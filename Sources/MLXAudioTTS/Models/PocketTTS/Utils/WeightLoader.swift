//
//  WeightLoader.swift
//  Swift-TTS
//
//  Weight loading utilities for PocketTTS models.
//

import Foundation
import MLX
import MLXNN
import Hub

// MARK: - Weight Sanitization

/// Sanitize weight keys for loading into Swift model
/// Handles differences between Python and Swift naming conventions
public func sanitizeWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
        var newKey = key
        var newValue = value

        // Transpose Conv1d and ConvTranspose1d weights from PyTorch format to MLX format
        //
        // Conv1d:
        //   PyTorch: [out_channels, in_channels, kernel_size]
        //   MLX: [out_channels, kernel_size, in_channels]
        //   Transform: swap axes 1 and 2
        //
        // ConvTranspose1d:
        //   PyTorch: [in_channels, out_channels, kernel_size]
        //   MLX: [out_channels, kernel_size, in_channels]
        //   Transform: permute [1, 2, 0]

        // Note: Mimi weights in smdesai/pocket-tts repo are pre-converted to MLX format
        // No transposition needed for mimi.* weights

        // Convert Python snake_case to Swift conventions
        // e.g., "flow_lm.transformer.layers.0.self_attn.in_proj.weight"
        //    -> "flowLM.transformer.layers.0.selfAttn.inProj.weight"

        // Handle common patterns
        newKey = newKey.replacingOccurrences(of: "flow_lm", with: "flowLM")

        // For Mimi weights, keep underscore naming for transformer properties
        // PocketTransformerLayer uses: self_attn, layer_scale_1, layer_scale_2
        // PocketAttention uses: in_proj, out_proj
        // PocketProjectedTransformer uses: input_proj, output_projs
        let isMimiTransformerKey = newKey.contains("mimi.") && (newKey.contains("_transformer.") || newKey.contains("Transformer."))

        if !isMimiTransformerKey {
            // Only convert to camelCase for non-Mimi transformer weights
            newKey = newKey.replacingOccurrences(of: "self_attn", with: "selfAttn")
            newKey = newKey.replacingOccurrences(of: "layer_scale", with: "layerScale")
            newKey = newKey.replacingOccurrences(of: "in_proj", with: "inProj")
            newKey = newKey.replacingOccurrences(of: "out_proj", with: "outProj")
            newKey = newKey.replacingOccurrences(of: "input_proj", with: "inputProj")
            newKey = newKey.replacingOccurrences(of: "output_proj", with: "outputProj")
        }
        newKey = newKey.replacingOccurrences(of: "out_norm", with: "outNorm")
        newKey = newKey.replacingOccurrences(of: "out_eos", with: "outEOS")
        newKey = newKey.replacingOccurrences(of: "input_linear", with: "inputLinear")
        newKey = newKey.replacingOccurrences(of: "cond_embed", with: "condEmbed")
        newKey = newKey.replacingOccurrences(of: "time_embed", with: "timeEmbed")
        newKey = newKey.replacingOccurrences(of: "res_blocks", with: "resBlocks")
        newKey = newKey.replacingOccurrences(of: "final_layer", with: "finalLayer")
        newKey = newKey.replacingOccurrences(of: "norm_final", with: "normFinal")
        newKey = newKey.replacingOccurrences(of: "emb_std", with: "embStd")
        newKey = newKey.replacingOccurrences(of: "emb_mean", with: "embMean")
        newKey = newKey.replacingOccurrences(of: "bos_emb", with: "bosEmb")
        newKey = newKey.replacingOccurrences(of: "speaker_proj_weight", with: "speakerProjWeight")
        newKey = newKey.replacingOccurrences(of: "lookup_table", with: "lookupTable")

        // Handle adaLN modulation
        newKey = newKey.replacingOccurrences(of: "adaLN_modulation", with: "adaLNModulation")
        newKey = newKey.replacingOccurrences(of: "in_ln", with: "inLN")

        // Handle flow network
        newKey = newKey.replacingOccurrences(of: "flow_net", with: "flowNet")

        // Handle mimi components - PocketMimiAdapter uses camelCase
        // (PocketMimi uses underscores but PocketTTSModel uses PocketMimiAdapter)
        newKey = newKey.replacingOccurrences(of: "encoder_transformer", with: "encoderTransformer")
        newKey = newKey.replacingOccurrences(of: "decoder_transformer", with: "decoderTransformer")

        // Note: Mimi transformer MLP paths already include "gating" in Python
        // (TransformerLayer.gating -> MlpNoGating.linear1/linear2)
        // No transformation needed - paths like layers.X.gating.linear1 are correct as-is

        // Transform mimi.upsample and mimi.downsample keys from Python's 2-level to Swift's 3-level structure
        // Python: mimi.upsample.convtr.convtr.weight (2 levels of convtr)
        // Swift:  mimi.upsample.convtr.convtr.convtr.weight (3 levels - PocketConvTrUpsample1d -> PocketStreamableConvTranspose1d -> PocketNormConvTranspose1d -> PocketConvTranspose1d)
        if newKey.hasPrefix("mimi.upsample.convtr.convtr.") && !newKey.contains("convtr.convtr.convtr.") {
            newKey = newKey.replacingOccurrences(of: "mimi.upsample.convtr.convtr.", with: "mimi.upsample.convtr.convtr.convtr.")
        }
        // Python: mimi.downsample.conv.conv.weight (2 levels of conv)
        // Swift:  mimi.downsample.conv.conv.conv.weight (3 levels - PocketConvDownsample1d -> PocketStreamableConv1d -> PocketNormConv1d -> PocketConv1d)
        if newKey.hasPrefix("mimi.downsample.conv.conv.") && !newKey.contains("conv.conv.conv.") {
            newKey = newKey.replacingOccurrences(of: "mimi.downsample.conv.conv.", with: "mimi.downsample.conv.conv.conv.")
        }

        // Transform Mimi decoder weights from Python Sequential structure to Swift named properties
        // Python: decoder.model.0.conv -> Swift: decoder.init_conv1d.conv.conv
        // Python: decoder.model.11.conv -> Swift: decoder.final_conv1d.conv.conv
        // Python: decoder.model.2.convtr -> Swift: decoder.layers.0.upsample.convtr.convtr
        // Python: decoder.model.3.block.X.conv -> Swift: decoder.layers.0.residuals.0.block.X-1.conv.conv
        // etc.
        if newKey.contains("mimi.decoder.model.") || newKey.contains("mimi.encoder.model.") {
            newKey = transformMimiModelKey(newKey)
        }

        // Handle Sequential layers - Python uses .0, .1 but MLX Sequential uses .layers.0, .layers.1
        // Transform adaLNModulation.0 -> adaLNModulation.layers.0
        // Transform adaLNModulation.1.weight -> adaLNModulation.layers.1.weight
        if let range = newKey.range(of: "adaLNModulation.") {
            let afterAdaLN = String(newKey[range.upperBound...])
            // Check if it starts with a digit
            if let firstChar = afterAdaLN.first, firstChar.isNumber {
                let prefix = String(newKey[..<range.upperBound])
                newKey = prefix + "layers." + afterAdaLN
            }
        }

        // Handle TimestepEmbedder mlp Sequential
        // Python: timeEmbed.0.mlp.2.weight -> Swift: timeEmbed.0.mlp.layers.2.weight
        // Pattern: timeEmbed.X.mlp.Y where X is embedder index and Y is layer index
        if newKey.contains("timeEmbed.") && newKey.contains(".mlp.") {
            // Find the mlp. part and add layers. after it
            if let mlpRange = newKey.range(of: ".mlp.") {
                let afterMlp = String(newKey[mlpRange.upperBound...])
                // Check if it starts with a digit
                if let firstChar = afterMlp.first, firstChar.isNumber {
                    let prefix = String(newKey[..<mlpRange.upperBound])
                    newKey = prefix + "layers." + afterMlp
                }
            }
        }

        // Handle ResBlock mlp Sequential -> named linear properties
        // Python: resBlocks.X.mlp.0.weight -> Swift: resBlocks.X.linear1.weight
        // Python: resBlocks.X.mlp.2.weight -> Swift: resBlocks.X.linear2.weight
        // The Python model uses Sequential([Linear, SiLU, Linear]) so index 0 is first linear, 2 is second
        if newKey.contains("resBlocks.") && newKey.contains(".mlp.") {
            newKey = newKey.replacingOccurrences(of: ".mlp.0.", with: ".linear1.")
            newKey = newKey.replacingOccurrences(of: ".mlp.2.", with: ".linear2.")
        }

        sanitized[newKey] = newValue
    }

    return sanitized
}

/// Transform Mimi encoder/decoder model keys from Python Sequential structure to Swift named properties
/// Python uses a flat Sequential: model.0, model.2, model.3, etc.
/// Swift uses named properties: init_conv1d, layers.0.upsample, layers.0.residuals.0, etc.
private func transformMimiModelKey(_ key: String) -> String {
    var result = key

    // Determine if it's encoder or decoder
    let isEncoder = key.contains("mimi.encoder.model.")
    let isDecoder = key.contains("mimi.decoder.model.")

    guard isEncoder || isDecoder else { return key }

    let prefix = isEncoder ? "mimi.encoder.model." : "mimi.decoder.model."

    // Extract the model index and rest of path
    guard let modelRange = result.range(of: prefix) else { return key }
    let afterModel = String(result[modelRange.upperBound...])

    // Parse the index: "0.conv.weight" -> index=0, rest=".conv.weight"
    guard let dotIndex = afterModel.firstIndex(of: ".") else { return key }
    let indexStr = String(afterModel[..<dotIndex])
    guard let modelIndex = Int(indexStr) else { return key }
    let rest = String(afterModel[dotIndex...])

    let basePrefix = isEncoder ? "mimi.encoder." : "mimi.decoder."

    if isDecoder {
        // Decoder structure (ratios [6, 5, 4] = 3 layers):
        // model.0 -> init_conv1d
        // model.2 -> layers.0.upsample (convtr)
        // model.3 -> layers.0.residuals.0 (block)
        // model.5 -> layers.1.upsample
        // model.6 -> layers.1.residuals.0
        // model.8 -> layers.2.upsample
        // model.9 -> layers.2.residuals.0
        // model.11 -> final_conv1d
        switch modelIndex {
        case 0:
            result = basePrefix + "init_conv1d" + transformConvPath(rest)
        case 2:
            result = basePrefix + "layers.0.upsample" + transformConvTrPath(rest)
        case 3:
            result = basePrefix + "layers.0.residuals.0" + transformBlockPath(rest)
        case 5:
            result = basePrefix + "layers.1.upsample" + transformConvTrPath(rest)
        case 6:
            result = basePrefix + "layers.1.residuals.0" + transformBlockPath(rest)
        case 8:
            result = basePrefix + "layers.2.upsample" + transformConvTrPath(rest)
        case 9:
            result = basePrefix + "layers.2.residuals.0" + transformBlockPath(rest)
        case 11:
            result = basePrefix + "final_conv1d" + transformConvPath(rest)
        default:
            break
        }
    } else {
        // Encoder structure (ratios reversed [4, 5, 6] for encoder):
        // model.0 -> init_conv1d
        // model.1 -> layers.0 (residuals + downsample)
        // model.3 -> layers.0.downsample
        // model.4 -> layers.1.residuals.0
        // model.6 -> layers.1.downsample
        // model.7 -> layers.2.residuals.0
        // model.9 -> layers.2.downsample
        // model.11 -> final_conv1d
        switch modelIndex {
        case 0:
            result = basePrefix + "init_conv1d" + transformConvPath(rest)
        case 1:
            result = basePrefix + "layers.0.residuals.0" + transformBlockPath(rest)
        case 3:
            result = basePrefix + "layers.0.downsample" + transformConvPath(rest)
        case 4:
            result = basePrefix + "layers.1.residuals.0" + transformBlockPath(rest)
        case 6:
            result = basePrefix + "layers.1.downsample" + transformConvPath(rest)
        case 7:
            result = basePrefix + "layers.2.residuals.0" + transformBlockPath(rest)
        case 9:
            result = basePrefix + "layers.2.downsample" + transformConvPath(rest)
        case 11:
            result = basePrefix + "final_conv1d" + transformConvPath(rest)
        default:
            break
        }
    }

    return result
}

/// Transform conv path: .conv.weight -> .conv.conv.weight
private func transformConvPath(_ path: String) -> String {
    // Python: .conv.weight -> Swift: .conv.conv.weight (StreamableConv1d -> NormConv1d -> Conv1d)
    return path.replacingOccurrences(of: ".conv.", with: ".conv.conv.")
}

/// Transform convtr path: .convtr.weight -> .convtr.convtr.weight
private func transformConvTrPath(_ path: String) -> String {
    // Python: .convtr.weight -> Swift: .convtr.convtr.weight
    return path.replacingOccurrences(of: ".convtr.", with: ".convtr.convtr.")
}

/// Transform block path for residual blocks
/// Python block uses: block.1.conv, block.3.conv (with ELU at 0, 2)
/// Swift block uses: block.0.conv, block.1.conv
private func transformBlockPath(_ path: String) -> String {
    var result = path

    // Transform block indices: block.1 -> block.0, block.3 -> block.1
    result = result.replacingOccurrences(of: ".block.1.", with: ".block.0.")
    result = result.replacingOccurrences(of: ".block.3.", with: ".block.1.")

    // Also transform the conv path within blocks
    result = result.replacingOccurrences(of: ".conv.", with: ".conv.conv.")

    return result
}

// MARK: - Weight Loading

/// Load weights from SafeTensors file
public func loadSafeTensorWeights(from url: URL) throws -> [String: MLXArray] {
    return try MLX.loadArrays(url: url)
}

/// Load weights from SafeTensors file with sanitization
public func loadAndSanitizeWeights(from url: URL) throws -> [String: MLXArray] {
    let weights = try loadSafeTensorWeights(from: url)
    return sanitizeWeightKeys(weights)
}

// MARK: - Model Weight Loading

/// Load weights into PocketTTS model
public func loadPocketTTSWeights(
    model: PocketTTSModel,
    from url: URL,
    strict: Bool = true
) throws {
    let weights = try loadAndSanitizeWeights(from: url)
    let parameters = ModuleParameters.unflattened(weights)

    if strict {
        try model.update(parameters: parameters, verify: .noUnusedKeys)
    } else {
        try model.update(parameters: parameters, verify: .none)
    }
}

/// Load weights from HuggingFace Hub
public func loadPocketTTSWeightsFromHub(
    model: PocketTTSModel,
    repoId: String,
    filename: String = "model.safetensors",
    progressHandler: @escaping @Sendable (Progress) -> Void
) async throws {
    let hub = HubApi.shared
    let repo = Hub.Repo(id: repoId)

    // Download snapshot
    let snapshotURL = try await hub.snapshot(from: repo, matching: [filename], progressHandler: progressHandler)
    let weightsURL = snapshotURL.appendingPathComponent(filename)

    // Load weights (non-strict to allow partial loading during development)
    try loadPocketTTSWeights(model: model, from: weightsURL, strict: false)
}

// MARK: - Separate Component Loading

/// Load only FlowLM weights
public func loadFlowLMWeights(
    model: FlowLMModel,
    from url: URL
) throws {
    let allWeights = try loadAndSanitizeWeights(from: url)

    // Filter to only flowLM weights
    var flowLMWeights: [String: MLXArray] = [:]
    for (key, value) in allWeights {
        if key.hasPrefix("flowLM.") {
            let newKey = String(key.dropFirst("flowLM.".count))
            flowLMWeights[newKey] = value
        }
    }

    let parameters = ModuleParameters.unflattened(flowLMWeights)
    try model.update(parameters: parameters, verify: .none)
}

/// Load only Mimi adapter weights
public func loadMimiAdapterWeights(
    adapter: PocketMimiAdapter,
    from url: URL
) throws {
    let allWeights = try loadAndSanitizeWeights(from: url)

    // Filter to mimi weights
    var mimiWeights: [String: MLXArray] = [:]
    for (key, value) in allWeights {
        if key.hasPrefix("mimi.") {
            let newKey = String(key.dropFirst("mimi.".count))
            mimiWeights[newKey] = value
        }
    }

    let parameters = ModuleParameters.unflattened(mimiWeights)
    try adapter.update(parameters: parameters, verify: .none)
}

// MARK: - Hub Repository Info

/// PocketTTS model repository information
public struct PocketTTSRepoInfo {
    /// Default HuggingFace repository containing model and tokenizer
    public static let defaultRepoId = "smdesai/pocket-tts"
    public static let defaultWeightsFile = "model.safetensors"
    public static let defaultConfigFile = "config.json"
    public static let defaultTokenizerFile = "tokenizer.json"
    public static let tokenizerConfigFile = "tokenizer_config.json"

    /// Voice embeddings repository (original Kyutai repo)
    public static let voiceRepoId = "kyutai/pocket-tts-without-voice-cloning"
    public static let voiceEmbeddingsPath = "embeddings"
}

// MARK: - Complete Model Loading

/// Load complete PocketTTS model from HuggingFace Hub (smdesai/pocket-tts)
/// - Parameters:
///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
///   - mimi: PocketMimi codec instance for audio decoding
///   - progressHandler: Download progress callback
/// - Returns: Initialized PocketTTSModel with loaded weights
public func loadPocketTTSFromHub(
    repoId: String = PocketTTSRepoInfo.defaultRepoId,
    mimi: PocketMimi,
    progressHandler: @escaping @Sendable (Progress) -> Void
) async throws -> PocketTTSModel {
    let hub = HubApi.shared
    let repo = Hub.Repo(id: repoId)

    // Download config, weights, and tokenizer
    let files = [
        PocketTTSRepoInfo.defaultConfigFile,
        PocketTTSRepoInfo.defaultWeightsFile,
        PocketTTSRepoInfo.defaultTokenizerFile,
        PocketTTSRepoInfo.tokenizerConfigFile
    ]

    let snapshotURL = try await hub.snapshot(from: repo, matching: files, progressHandler: progressHandler)

    // Load config
    let configURL = snapshotURL.appendingPathComponent(PocketTTSRepoInfo.defaultConfigFile)
    let config = try PocketTTSModelConfig.load(from: configURL)

    // Create model (tokenizer will be loaded from the same repo)
    let model = try await PocketTTSModel.fromConfig(
        config,
        mimi: mimi,
        repoId: repoId,
        progressHandler: progressHandler
    )

    // Load weights
    let weightsURL = snapshotURL.appendingPathComponent(PocketTTSRepoInfo.defaultWeightsFile)
    try loadPocketTTSWeights(model: model, from: weightsURL, strict: false)

    return model
}
