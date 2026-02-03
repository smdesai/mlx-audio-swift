//
//  PocketTTSConfig.swift
//  Swift-TTS
//
//  PocketTTS configuration structures for loading model parameters from JSON/YAML.
//

import Foundation
import Hub

// MARK: - Flow Network Configuration

public struct FlowConfig: Codable, Sendable {
    public let dim: Int
    public let depth: Int

    public init(dim: Int, depth: Int) {
        self.dim = dim
        self.depth = depth
    }
}

// MARK: - FlowLM Transformer Configuration

public struct FlowLMTransformerConfig: Codable, Sendable {
    public let hiddenScale: Int
    public let maxPeriod: Int
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int

    enum CodingKeys: String, CodingKey {
        case hiddenScale = "hidden_scale"
        case maxPeriod = "max_period"
        case dModel = "d_model"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
    }

    public init(hiddenScale: Int, maxPeriod: Int, dModel: Int, numHeads: Int, numLayers: Int) {
        self.hiddenScale = hiddenScale
        self.maxPeriod = maxPeriod
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
    }
}

// MARK: - Lookup Table Configuration

public struct LookupTableConfig: Codable, Sendable {
    public let dim: Int
    public let nBins: Int
    public let tokenizer: String
    public let tokenizerPath: String

    enum CodingKeys: String, CodingKey {
        case dim
        case nBins = "n_bins"
        case tokenizer
        case tokenizerPath = "tokenizer_path"
    }

    public init(dim: Int, nBins: Int, tokenizer: String, tokenizerPath: String) {
        self.dim = dim
        self.nBins = nBins
        self.tokenizer = tokenizer
        self.tokenizerPath = tokenizerPath
    }
}

// MARK: - FlowLM Configuration

public struct FlowLMConfig: Codable, Sendable {
    public let dtype: String?
    public let flow: FlowConfig
    public let transformer: FlowLMTransformerConfig
    public let lookupTable: LookupTableConfig
    public let weightsPath: String?

    enum CodingKeys: String, CodingKey {
        case dtype
        case flow
        case transformer
        case lookupTable = "lookup_table"
        case weightsPath = "weights_path"
    }

    public init(
        dtype: String? = nil,
        flow: FlowConfig,
        transformer: FlowLMTransformerConfig,
        lookupTable: LookupTableConfig,
        weightsPath: String? = nil
    ) {
        self.dtype = dtype
        self.flow = flow
        self.transformer = transformer
        self.lookupTable = lookupTable
        self.weightsPath = weightsPath
    }
}

// MARK: - SEANet Configuration

public struct SEANetConfig: Codable, Sendable {
    public let dimension: Int
    public let channels: Int
    public let nFilters: Int
    public let nResidualLayers: Int
    public let ratios: [Int]
    public let kernelSize: Int
    public let residualKernelSize: Int
    public let lastKernelSize: Int
    public let dilationBase: Int
    public let padMode: String
    public let compress: Int

    enum CodingKeys: String, CodingKey {
        case dimension
        case channels
        case nFilters = "n_filters"
        case nResidualLayers = "n_residual_layers"
        case ratios
        case kernelSize = "kernel_size"
        case residualKernelSize = "residual_kernel_size"
        case lastKernelSize = "last_kernel_size"
        case dilationBase = "dilation_base"
        case padMode = "pad_mode"
        case compress
    }

    public init(
        dimension: Int,
        channels: Int,
        nFilters: Int,
        nResidualLayers: Int,
        ratios: [Int],
        kernelSize: Int,
        residualKernelSize: Int,
        lastKernelSize: Int,
        dilationBase: Int,
        padMode: String,
        compress: Int
    ) {
        self.dimension = dimension
        self.channels = channels
        self.nFilters = nFilters
        self.nResidualLayers = nResidualLayers
        self.ratios = ratios
        self.kernelSize = kernelSize
        self.residualKernelSize = residualKernelSize
        self.lastKernelSize = lastKernelSize
        self.dilationBase = dilationBase
        self.padMode = padMode
        self.compress = compress
    }
}

// MARK: - Mimi Transformer Configuration

public struct MimiTransformerConfig: Codable, Sendable {
    public let dModel: Int
    public let inputDimension: Int
    public let outputDimensions: [Int]
    public let numHeads: Int
    public let numLayers: Int
    public let layerScale: Float
    public let context: Int
    public let dimFeedforward: Int
    public let maxPeriod: Float

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case inputDimension = "input_dimension"
        case outputDimensions = "output_dimensions"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case layerScale = "layer_scale"
        case context
        case dimFeedforward = "dim_feedforward"
        case maxPeriod = "max_period"
    }

    public init(
        dModel: Int,
        inputDimension: Int,
        outputDimensions: [Int],
        numHeads: Int,
        numLayers: Int,
        layerScale: Float,
        context: Int,
        dimFeedforward: Int,
        maxPeriod: Float = 10000.0
    ) {
        self.dModel = dModel
        self.inputDimension = inputDimension
        self.outputDimensions = outputDimensions
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.layerScale = layerScale
        self.context = context
        self.dimFeedforward = dimFeedforward
        self.maxPeriod = maxPeriod
    }
}

// MARK: - Quantizer Configuration

public struct QuantizerConfig: Codable, Sendable {
    public let dimension: Int
    public let outputDimension: Int

    enum CodingKeys: String, CodingKey {
        case dimension
        case outputDimension = "output_dimension"
    }

    public init(dimension: Int, outputDimension: Int) {
        self.dimension = dimension
        self.outputDimension = outputDimension
    }
}

// MARK: - Mimi Configuration (JSON parsing for PocketTTS config.json)

public struct PocketMimiJSONConfig: Codable, Sendable {
    public let dtype: String?
    public let sampleRate: Int
    public let channels: Int
    public let frameRate: Float
    public let seanet: SEANetConfig
    public let transformer: MimiTransformerConfig
    public let quantizer: QuantizerConfig
    public let weightsPath: String?

    enum CodingKeys: String, CodingKey {
        case dtype
        case sampleRate = "sample_rate"
        case channels
        case frameRate = "frame_rate"
        case seanet
        case transformer
        case quantizer
        case weightsPath = "weights_path"
    }

    public init(
        dtype: String? = nil,
        sampleRate: Int,
        channels: Int,
        frameRate: Float,
        seanet: SEANetConfig,
        transformer: MimiTransformerConfig,
        quantizer: QuantizerConfig,
        weightsPath: String? = nil
    ) {
        self.dtype = dtype
        self.sampleRate = sampleRate
        self.channels = channels
        self.frameRate = frameRate
        self.seanet = seanet
        self.transformer = transformer
        self.quantizer = quantizer
        self.weightsPath = weightsPath
    }
}

// MARK: - PocketTTS Model Configuration

public struct PocketTTSModelConfig: Codable, Sendable {
    public let modelType: String
    public let flowLM: FlowLMConfig?
    public let mimi: PocketMimiJSONConfig?
    public let weightsPath: String?
    public let weightsPathWithoutVoiceCloning: String?
    public let modelPath: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case flowLM = "flow_lm"
        case mimi
        case weightsPath = "weights_path"
        case weightsPathWithoutVoiceCloning = "weights_path_without_voice_cloning"
        case modelPath = "model_path"
    }

    public init(
        modelType: String = "pocket_tts",
        flowLM: FlowLMConfig? = nil,
        mimi: PocketMimiJSONConfig? = nil,
        weightsPath: String? = nil,
        weightsPathWithoutVoiceCloning: String? = nil,
        modelPath: String? = nil
    ) {
        self.modelType = modelType
        self.flowLM = flowLM
        self.mimi = mimi
        self.weightsPath = weightsPath
        self.weightsPathWithoutVoiceCloning = weightsPathWithoutVoiceCloning
        self.modelPath = modelPath
    }

    // MARK: - Configuration Loading

    public static func load(from url: URL) throws -> PocketTTSModelConfig {
        let data = try Data(contentsOf: url)

        if url.pathExtension.lowercased() == "yaml" || url.pathExtension.lowercased() == "yml" {
            // For YAML, we'd need a YAML parser - for now, assume JSON
            throw PocketTTSError.unsupportedFormat("YAML parsing not yet implemented. Use JSON.")
        }

        let decoder = JSONDecoder()
        return try decoder.decode(PocketTTSModelConfig.self, from: data)
    }

    public static func load(from path: String) throws -> PocketTTSModelConfig {
        return try load(from: URL(fileURLWithPath: path))
    }

    /// Load configuration from HuggingFace Hub
    /// - Parameters:
    ///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
    ///   - progressHandler: Optional download progress callback
    /// - Returns: Loaded configuration
    public static func loadFromHub(
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> PocketTTSModelConfig {
        let hub = HubApi.shared
        let repo = Hub.Repo(id: repoId)

        let handler: (Progress) -> Void = progressHandler ?? { _ in }
        let snapshotURL = try await hub.snapshot(
            from: repo,
            matching: [PocketTTSRepo.configFile],
            progressHandler: handler
        )

        let configURL = snapshotURL.appendingPathComponent(PocketTTSRepo.configFile)
        return try load(from: configURL)
    }
}

// MARK: - Errors

public enum PocketTTSError: Error, LocalizedError {
    case configurationMissing(String)
    case unsupportedFormat(String)
    case weightLoadingFailed(String)
    case tokenizerError(String)
    case generationError(String)

    public var errorDescription: String? {
        switch self {
        case .configurationMissing(let component):
            return "Missing configuration for: \(component)"
        case .unsupportedFormat(let format):
            return "Unsupported format: \(format)"
        case .weightLoadingFailed(let reason):
            return "Failed to load weights: \(reason)"
        case .tokenizerError(let reason):
            return "Tokenizer error: \(reason)"
        case .generationError(let reason):
            return "Generation error: \(reason)"
        }
    }
}

// MARK: - Generation Defaults

public struct PocketTTSDefaults {
    public static let temperature: Float = 0.7
    public static let lsdDecodeSteps: Int = 1
    public static let noiseClamp: Float? = nil
    // EOS threshold: matches Python DEFAULT_EOS_THRESHOLD = -4.0
    // During generation, EOS logits are typically around -10 (well below threshold)
    // EOS triggers when logit > threshold (i.e., when model signals end)
    public static let eosThreshold: Float = -4.0
    // Minimum frames to generate before EOS can trigger (prevents early cutoff)
    public static let minStepsBeforeEOS: Int = 10
    public static let defaultVoice: String = "alba"
}
