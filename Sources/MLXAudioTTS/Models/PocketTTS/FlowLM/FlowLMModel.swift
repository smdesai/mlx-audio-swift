//
//  FlowLMModel.swift
//  Swift-TTS
//
//  Flow Language Model combining transformer backbone with flow matching for PocketTTS.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - FlowLM Model

/// Flow Language Model that combines:
/// - LUT Conditioner for text embeddings
/// - Streaming Transformer backbone
/// - Flow MLP for latent prediction via LSD decode
public class FlowLMModel: Module {
    public let conditioner: LUTConditioner
    public let ldim: Int  // Latent dimension
    public let dim: Int   // Model dimension
    public let statsEMADecay: Float
    public let textPaddingWeight: Float

    private let flowNet: SimpleMLPAdaLN
    private let inputLinear: Linear
    private let transformer: StreamingTransformer
    private let outNorm: LayerNorm
    public let outEOS: Linear  // Public for debugging weight loading

    // Learned parameters
    public var embStd: MLXArray
    public var embMean: MLXArray
    public var bosEmb: MLXArray

    public init(
        conditioner: LUTConditioner,
        flowNet: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: Int = 128,
        ldim: Int = 64,
        statsEMADecay: Float = 0.999,
        textPaddingWeight: Float = 1.0,
        dtype: DType? = nil
    ) {
        self.conditioner = conditioner
        self.ldim = ldim
        self.statsEMADecay = statsEMADecay
        self.dim = dim
        self.textPaddingWeight = textPaddingWeight

        self.flowNet = flowNet

        // Initialize statistics
        let actualDtype = dtype ?? .float32
        self.embStd = MLXArray.ones([ldim]).asType(actualDtype)
        self.embMean = MLXArray.zeros([ldim]).asType(actualDtype)
        self.bosEmb = MLXRandom.normal([ldim]).asType(actualDtype)

        // Layers
        self.inputLinear = Linear(ldim, dim, bias: false)
        self.transformer = transformer
        self.outNorm = LayerNorm(dimensions: dim, eps: 1e-5)
        self.outEOS = Linear(dim, 1)

        super.init()
    }

    // MARK: - Cache Management

    /// Create KV caches for all transformer layers
    public func makeCache() -> [KVCacheSimple] {
        return transformer.makeCache()
    }

    // MARK: - Backbone Forward

    /// Run transformer backbone
    /// - Parameters:
    ///   - input: Input latent embeddings [B, T, ldim]
    ///   - textEmbeddings: Text embeddings from conditioner [B, textLen, dim]
    ///   - sequence: Original sequence for shape reference [B, seqLen, ldim]
    ///   - cache: KV caches for streaming
    /// - Returns: Transformer output for sequence positions [B, seqLen, dim]
    public func backbone(
        input: MLXArray,
        textEmbeddings: MLXArray,
        sequence: MLXArray,
        cache: [KVCacheSimple]?
    ) -> MLXArray {
        // Concatenate text embeddings with input
        let combined = MLX.concatenated([textEmbeddings, input], axis: 1)

        // Run through transformer
        var transformerOut = transformer(combined, cache: cache)

        // Apply output norm
        transformerOut = outNorm(transformerOut)

        // Return only the sequence portion (after text)
        // When seqLen is 0, return entire output (matches Python: transformer_out[:, -0:] = entire array)
        let seqLen = sequence.shape[1]
        if seqLen == 0 {
            return transformerOut
        }
        let totalLen = transformerOut.shape[1]
        return transformerOut[0..., (totalLen - seqLen)..., 0...]
    }

    // MARK: - Forward Pass (Generation)

    /// Generate next latent using flow matching
    /// - Parameters:
    ///   - sequence: Current sequence of latents [B, T, ldim]
    ///   - textEmbeddings: Text embeddings [B, textLen, dim]
    ///   - cache: KV caches
    ///   - lsdDecodeSteps: Number of LSD decode steps
    ///   - temp: Temperature for noise sampling
    ///   - noiseClamp: Optional noise clamping
    ///   - eosThreshold: Threshold for EOS detection
    /// - Returns: (nextLatent [B, ldim], isEOS [B])
    public func callAsFunction(
        sequence: MLXArray,
        textEmbeddings: MLXArray,
        cache: [KVCacheSimple]?,
        lsdDecodeSteps: Int,
        temp: Float,
        noiseClamp: Float?,
        eosThreshold: Float
    ) -> (MLXArray, MLXArray) {
        precondition(lsdDecodeSteps > 0, "lsdDecodeSteps must be > 0 for generation.")

        // Replace NaN with BOS embedding
        let bos = bosEmb.reshaped([1, 1, ldim])
        let nanMask = sequence .!= sequence  // NaN != NaN is true
        let maskedSequence = MLX.where(nanMask, bos, sequence)

        // Project input
        let input = inputLinear(maskedSequence)

        // Get transformer output
        var transformerOut = backbone(
            input: input,
            textEmbeddings: textEmbeddings,
            sequence: maskedSequence,
            cache: cache
        )

        // Cast to float32 for precision in flow matching
        transformerOut = transformerOut.asType(DType.float32)

        // Take last position for generation
        let lastOut = transformerOut[0..., -1, 0...]  // [B, dim]

        // Check for EOS
        let outEOSLogit = outEOS(lastOut)
        let isEOS = MLX.greater(outEOSLogit, MLXArray(eosThreshold))

        // Sample noise
        let noiseShape = [lastOut.shape[0], ldim]
        let std = sqrt(temp)
        var noise = MLXRandom.normal(noiseShape).asType(transformerOut.dtype) * std

        // Optionally clamp noise
        if let clamp = noiseClamp {
            noise = MLX.clip(noise, min: -clamp, max: clamp)
        }

        // LSD decode with flow network
        let noiseExpanded = noise.reshaped([noise.shape[0], 1, ldim])
        let decoded = lsdDecodeConditioned(
            flowNet: flowNet,
            conditioning: lastOut,
            x0: noiseExpanded,
            numSteps: lsdDecodeSteps
        )

        // Squeeze back: [B, 1, ldim] -> [B, ldim]
        let nextLatent = decoded.squeezed(axis: 1)

        return (nextLatent, isEOS.squeezed())
    }

    // MARK: - Sample Next Latent

    /// Convenience method for sampling next latent
    public func sampleNextLatent(
        sequence: MLXArray,
        textEmbeddings: MLXArray,
        cache: [KVCacheSimple]?,
        lsdDecodeSteps: Int,
        temp: Float,
        noiseClamp: Float?,
        eosThreshold: Float
    ) -> (MLXArray, MLXArray) {
        return self(
            sequence: sequence,
            textEmbeddings: textEmbeddings,
            cache: cache,
            lsdDecodeSteps: lsdDecodeSteps,
            temp: temp,
            noiseClamp: noiseClamp,
            eosThreshold: eosThreshold
        )
    }

    // MARK: - Factory Methods

    /// Create FlowLMModel from configuration, loading tokenizer from HuggingFace
    /// - Parameters:
    ///   - config: FlowLM configuration
    ///   - latentDim: Latent dimension for audio codec
    ///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
    ///   - progressHandler: Optional download progress callback
    public static func fromConfig(
        _ config: FlowLMConfig,
        latentDim: Int,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> FlowLMModel {
        let dModel = config.transformer.dModel

        // Create flow MLP
        let flowMLP = SimpleMLPAdaLN.fromConfig(config, latentDim: latentDim, condDim: dModel)

        // Create conditioner with tokenizer from HuggingFace
        let conditioner = try await LUTConditioner.fromConfig(
            config: config.lookupTable,
            outputDim: dModel,
            repoId: repoId,
            progressHandler: progressHandler
        )

        // Create transformer
        let transformer = StreamingTransformer.fromConfig(config.transformer)

        // Parse dtype
        let dtype: DType? = config.dtype.flatMap { dtypeString in
            switch dtypeString.lowercased() {
            case "float32": return .float32
            case "float16": return .float16
            case "bfloat16": return .bfloat16
            default: return nil
            }
        }

        return FlowLMModel(
            conditioner: conditioner,
            flowNet: flowMLP,
            transformer: transformer,
            dim: dModel,
            ldim: latentDim,
            dtype: dtype
        )
    }

    /// Create FlowLMModel from local model folder
    public static func fromLocalFolder(
        _ config: FlowLMConfig,
        latentDim: Int,
        modelFolder: URL
    ) throws -> FlowLMModel {
        let dModel = config.transformer.dModel

        let flowMLP = SimpleMLPAdaLN.fromConfig(config, latentDim: latentDim, condDim: dModel)

        let conditioner = try LUTConditioner.fromLocalFolder(
            config: config.lookupTable,
            outputDim: dModel,
            modelFolder: modelFolder
        )

        let transformer = StreamingTransformer.fromConfig(config.transformer)

        let dtype: DType? = config.dtype.flatMap { dtypeString in
            switch dtypeString.lowercased() {
            case "float32": return .float32
            case "float16": return .float16
            case "bfloat16": return .bfloat16
            default: return nil
            }
        }

        return FlowLMModel(
            conditioner: conditioner,
            flowNet: flowMLP,
            transformer: transformer,
            dim: dModel,
            ldim: latentDim,
            dtype: dtype
        )
    }
}
