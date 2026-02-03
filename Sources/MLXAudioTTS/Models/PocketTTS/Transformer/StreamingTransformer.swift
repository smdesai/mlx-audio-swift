//
//  StreamingTransformer.swift
//  Swift-TTS
//
//  Streaming transformer layers for PocketTTS backbone.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Streaming Transformer Layer

/// Single transformer layer with streaming support
public class StreamingTransformerLayer: Module {
    private let selfAttn: StreamingMultiheadAttention
    private let norm1: LayerNorm
    private let norm2: LayerNorm
    private let linear1: Linear
    private let linear2: Linear
    private let layerScale1: PocketCausalLayerScale?
    private let layerScale2: PocketCausalLayerScale?

    public init(
        dModel: Int,
        numHeads: Int,
        dimFeedforward: Int,
        rope: RotaryEmbedding,
        layerScale: Float? = nil
    ) {
        self.selfAttn = StreamingMultiheadAttention(
            embedDim: dModel,
            numHeads: numHeads,
            rope: rope
        )

        self.norm1 = LayerNorm(dimensions: dModel, eps: 1e-5)
        self.norm2 = LayerNorm(dimensions: dModel, eps: 1e-5)

        self.linear1 = Linear(dModel, dimFeedforward, bias: false)
        self.linear2 = Linear(dimFeedforward, dModel, bias: false)

        if let scale = layerScale {
            self.layerScale1 = PocketCausalLayerScale(channels: dModel, initValue: scale)
            self.layerScale2 = PocketCausalLayerScale(channels: dModel, initValue: scale)
        } else {
            self.layerScale1 = nil
            self.layerScale2 = nil
        }

        super.init()
    }

    private func applyScale(_ x: MLXArray, layerScale: PocketCausalLayerScale?) -> MLXArray {
        if let scale = layerScale {
            return scale(x)
        }
        return x
    }

    public func callAsFunction(_ x: MLXArray, cache: KVCacheSimple? = nil) -> MLXArray {
        // Self-attention with residual
        let attnOut = selfAttn(norm1(x), cache: cache)
        var out = x + applyScale(attnOut, layerScale: layerScale1)

        // Feedforward with residual
        let ffOut = linear2(gelu(linear1(norm2(out))))
        out = out + applyScale(ffOut, layerScale: layerScale2)

        return out
    }
}

// MARK: - Streaming Transformer

/// Multi-layer streaming transformer
public class StreamingTransformer: Module {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let headDim: Int

    private let rope: RotaryEmbedding
    private let layers: [StreamingTransformerLayer]

    public init(
        dModel: Int,
        numHeads: Int,
        numLayers: Int,
        dimFeedforward: Int,
        maxPeriod: Float = 10000.0,
        layerScale: Float? = nil
    ) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.headDim = dModel / numHeads

        let ropeInstance = RotaryEmbedding(maxPeriod: maxPeriod)
        self.rope = ropeInstance

        self.layers = (0..<numLayers).map { _ in
            StreamingTransformerLayer(
                dModel: dModel,
                numHeads: numHeads,
                dimFeedforward: dimFeedforward,
                rope: ropeInstance,
                layerScale: layerScale
            )
        }

        super.init()
    }

    /// Forward pass through all transformer layers
    /// - Parameters:
    ///   - x: Input tensor of shape [B, T, D]
    ///   - cache: Optional list of KV caches (one per layer)
    /// - Returns: Output tensor of shape [B, T, D]
    public func callAsFunction(_ x: MLXArray, cache: [KVCacheSimple]? = nil) -> MLXArray {
        var out = x

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[safe: i]
            out = layer(out, cache: layerCache)
        }

        return out
    }

    /// Create KV caches for all layers
    public func makeCache() -> [KVCacheSimple] {
        return (0..<numLayers).map { _ in
            KVCacheSimple()
        }
    }

    /// Create from config
    public static func fromConfig(_ config: FlowLMTransformerConfig) -> StreamingTransformer {
        return StreamingTransformer(
            dModel: config.dModel,
            numHeads: config.numHeads,
            numLayers: config.numLayers,
            dimFeedforward: config.hiddenScale * config.dModel,
            maxPeriod: Float(config.maxPeriod)
        )
    }
}

// MARK: - Array Safe Access Extension

extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}
