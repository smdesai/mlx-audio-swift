//
//  MimiAdapter.swift
//  Swift-TTS
//
//  Adapter for Mimi codec in PocketTTS, using continuous latents instead of discrete codes.
//

import Foundation
import MLX
import MLXNN

// MARK: - Dummy Quantizer

/// Simple projection layer that replaces the quantizer for continuous latent space
/// PocketTTS works with continuous latents, not discrete codes
public class DummyQuantizer: Module {
    // Use camelCase key to match sanitized weight keys (output_proj -> outputProj)
    @ModuleInfo public var outputProj: PocketConv1d
    private let inputDim: Int
    private let outputDim: Int

    public init(dimension: Int, outputDimension: Int) {
        self.inputDim = dimension
        self.outputDim = outputDimension
        // Use PocketConv1d which handles NCL format (like PyTorch)
        // Input: [B, inChannels, L] -> Output: [B, outChannels, L]
        self._outputProj = ModuleInfo(wrappedValue: PocketConv1d(
            inChannels: dimension,
            outChannels: outputDimension,
            ksize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            dilation: 1,
            bias: false
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return outputProj(x)
    }
}

// MARK: - Mimi Adapter for PocketTTS

/// Adapter that wraps the Marvis Mimi codec for PocketTTS
/// Uses continuous latent space instead of discrete codes
public class PocketMimiAdapter: Module, @unchecked Sendable {
    // Core components (reused from Marvis Mimi if available)
    public let frameRate: Float
    public let encoderFrameRate: Float
    public let sampleRate: Int
    public let channels: Int
    public let dimension: Int

    // Quantizer replacement for continuous latents
    @ModuleInfo public var quantizer: DummyQuantizer

    // Mimi codec components exposed via @ModuleInfo for weight loading
    // Weight keys like mimi.decoder.* will map to these properties
    @ModuleInfo public var encoder: PocketSeanetEncoder
    @ModuleInfo public var decoder: PocketSeanetDecoder
    @ModuleInfo public var encoderTransformer: PocketProjectedTransformer
    @ModuleInfo public var decoderTransformer: PocketProjectedTransformer
    @ModuleInfo public var downsample: PocketConvDownsample1d
    @ModuleInfo public var upsample: PocketConvTrUpsample1d

    // Caches for streaming
    public var encoderCache: [PocketKVCache]
    public var decoderCache: [PocketKVCache]

    public init(
        frameRate: Float,
        encoderFrameRate: Float,
        sampleRate: Int,
        channels: Int,
        dimension: Int,
        outputDimension: Int
    ) {
        self.frameRate = frameRate
        self.encoderFrameRate = encoderFrameRate
        self.sampleRate = sampleRate
        self.channels = channels
        self.dimension = dimension

        // Create default Mimi config for components
        let cfg = pocketMimiConfig()
        let downsampleStride = Int(cfg.sampleRate / cfg.frameRate / Double(cfg.seanet.ratios.reduce(1, *)))

        self._quantizer = ModuleInfo(wrappedValue: DummyQuantizer(dimension: dimension, outputDimension: outputDimension))
        self._encoder = ModuleInfo(wrappedValue: PocketSeanetEncoder(cfg: cfg.seanet))
        self._decoder = ModuleInfo(wrappedValue: PocketSeanetDecoder(cfg: cfg.seanet))
        self._encoderTransformer = ModuleInfo(wrappedValue: PocketProjectedTransformer(
            cfg: cfg.transformer,
            inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]
        ))
        self._decoderTransformer = ModuleInfo(wrappedValue: PocketProjectedTransformer(
            cfg: cfg.transformer,
            inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]
        ))
        self._downsample = ModuleInfo(wrappedValue: PocketConvDownsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true
        ))
        self._upsample = ModuleInfo(wrappedValue: PocketConvTrUpsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true
        ))

        self.encoderCache = _encoderTransformer.wrappedValue.makeCache()
        self.decoderCache = _decoderTransformer.wrappedValue.makeCache()

        super.init()
    }

    /// Initialize with existing PocketMimi codec
    /// - Parameters:
    ///   - mimi: The PocketMimi codec instance
    ///   - inputDimension: Input dimension from FlowLM latent space (ldim)
    ///   - outputDimension: Output dimension for Mimi decoder
    public init(mimi: PocketMimi, inputDimension: Int, outputDimension: Int) {
        self.frameRate = Float(mimi.cfg.frameRate)
        self.encoderFrameRate = Float(mimi.cfg.sampleRate) / Float(mimi.cfg.seanet.ratios.reduce(1, *))
        self.sampleRate = Int(mimi.cfg.sampleRate)
        self.channels = mimi.cfg.channels
        self.dimension = mimi.cfg.seanet.dimension

        // Quantizer projects from FlowLM latent dim to Mimi's expected input
        self._quantizer = ModuleInfo(wrappedValue: DummyQuantizer(dimension: inputDimension, outputDimension: outputDimension))

        // Use the mimi's components directly (they will be replaced by weight loading)
        self._encoder = ModuleInfo(wrappedValue: mimi.encoder)
        self._decoder = ModuleInfo(wrappedValue: mimi.decoder)
        self._encoderTransformer = ModuleInfo(wrappedValue: mimi.encoder_transformer)
        self._decoderTransformer = ModuleInfo(wrappedValue: mimi.decoder_transformer)
        self._downsample = ModuleInfo(wrappedValue: mimi.downsample)
        self._upsample = ModuleInfo(wrappedValue: mimi.upsample)

        self.encoderCache = mimi.encoderCache
        self.decoderCache = mimi.decoderCache

        super.init()
    }

    /// Frame size in samples
    public var frameSize: Int {
        return Int(Float(sampleRate) / frameRate)
    }

    // MARK: - State Management

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
        upsample.resetState()
        downsample.resetState()
        for cache in encoderCache {
            cache.reset()
        }
        for cache in decoderCache {
            cache.reset()
        }
    }

    // MARK: - Encoding (Audio -> Latent)

    /// Encode audio to continuous latent representation
    /// - Parameter x: Audio tensor [B, C, T]
    /// - Returns: Latent tensor [B, D, Tlatent]
    public func encodeToLatent(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 3, "MimiAdapter.encodeToLatent expects audio of shape [B, C, T]")

        // Reset state for fresh encoding
        encoder.resetState()
        for cache in encoderCache {
            cache.reset()
        }

        // Pad for frame alignment
        let frameSize = self.frameSize
        let x_padded = padForConv1d(x, kernelSize: frameSize, stride: frameSize)

        // Encode through SEANet encoder
        var emb = encoder(x_padded)

        // Transform through encoder transformer
        emb = encoderTransformer(emb, cache: encoderCache)[0]

        // Downsample to target frame rate
        emb = downsample(emb)

        return emb
    }

    // MARK: - Decoding (Latent -> Audio)

    /// Decode continuous latent to audio (full sequence)
    /// - Parameter latent: Latent tensor [B, D, T]
    /// - Returns: Audio tensor [B, C, Taudio]
    public func decodeFromLatent(_ latent: MLXArray) -> MLXArray {
        // Reset state
        decoder.resetState()
        for cache in decoderCache {
            cache.reset()
        }
        upsample.resetState()

        // Upsample to encoder frame rate
        var emb = upsample(latent)

        // Transform through decoder transformer
        emb = decoderTransformer(emb, cache: decoderCache)[0]

        // Decode through SEANet decoder
        return decoder(emb)
    }

    /// Decode single step for streaming
    /// - Parameter latent: Single latent frame [B, D, 1]
    /// - Returns: Audio samples for this frame
    public func decodeStep(_ latent: MLXArray) -> MLXArray {
        // Upsample
        var emb = upsample.step(latent)

        // Transform
        emb = decoderTransformer(emb, cache: decoderCache)[0]

        // Decode
        return decoder.step(emb)
    }

    // MARK: - Factory

    /// Create adapter from PocketTTS Mimi config
    public static func fromConfig(_ config: PocketTTSMimiConfig, outputDimension: Int) -> PocketMimiAdapter {
        return PocketMimiAdapter(
            frameRate: Float(config.frameRate),
            encoderFrameRate: Float(config.sampleRate) / Float(config.seanet.ratios.reduce(1, *)),
            sampleRate: Int(config.sampleRate),
            channels: config.channels,
            dimension: config.transformer.dModel,
            outputDimension: outputDimension
        )
    }

    /// Create adapter wrapping existing PocketMimi codec
    /// - Parameters:
    ///   - mimi: The PocketMimi codec
    ///   - inputDimension: Input dimension from FlowLM latent space (ldim)
    ///   - outputDimension: Output dimension for Mimi decoder
    public static func fromMimi(_ mimi: PocketMimi, inputDimension: Int, outputDimension: Int) -> PocketMimiAdapter {
        return PocketMimiAdapter(mimi: mimi, inputDimension: inputDimension, outputDimension: outputDimension)
    }
}

// MARK: - Config Alias

public typealias PocketTTSMimiConfig = PocketMimiConfig

// MARK: - Padding Utility

/// Pad tensor for conv1d alignment
func padForConv1d(_ x: MLXArray, kernelSize: Int, stride: Int, paddingTotal: Int = 0) -> MLXArray {
    let length = x.shape[2]
    let extraPadding = getExtraPaddingForConv1d(length: length, kernelSize: kernelSize, stride: stride, paddingTotal: paddingTotal)

    if extraPadding <= 0 {
        return x
    }

    // Pad on the right along time dimension
    return MLX.padded(x, widths: [[0, 0], [0, 0], [0, extraPadding]])
}

/// Calculate extra padding needed for conv1d
func getExtraPaddingForConv1d(length: Int, kernelSize: Int, stride: Int, paddingTotal: Int) -> Int {
    let nFrames = (length + paddingTotal - kernelSize + stride) / stride
    let idealLength = (nFrames - 1) * stride + kernelSize - paddingTotal

    return max(0, idealLength - length)
}

