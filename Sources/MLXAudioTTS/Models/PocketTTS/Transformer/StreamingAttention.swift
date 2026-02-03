//
//  StreamingAttention.swift
//  Swift-TTS
//
//  Streaming multi-head attention with RoPE for PocketTTS.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Streaming Multihead Attention

/// Streaming multi-head attention with rotary position embeddings and KV cache support
public class StreamingMultiheadAttention: Module {
    public let embedDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rope: RotaryEmbedding?

    private let inProj: Linear
    private let outProj: Linear

    public init(
        embedDim: Int,
        numHeads: Int,
        rope: RotaryEmbedding? = nil
    ) {
        precondition(embedDim % numHeads == 0, "embed_dim must be divisible by num_heads")

        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = rope

        // Combined QKV projection (no bias)
        self.inProj = Linear(embedDim, 3 * embedDim, bias: false)
        self.outProj = Linear(embedDim, embedDim, bias: false)

        super.init()
    }

    /// Forward pass with optional KV cache for streaming
    /// - Parameters:
    ///   - query: Input tensor of shape [B, T, D]
    ///   - cache: Optional KV cache for streaming
    /// - Returns: Output tensor of shape [B, T, D]
    public func callAsFunction(_ query: MLXArray, cache: KVCacheSimple? = nil) -> MLXArray {
        let b = query.shape[0]
        let t = query.shape[1]

        // Project to QKV
        let projected = inProj(query)
        let projReshaped = projected.reshaped([b, t, 3, numHeads, headDim])

        var q = projReshaped[0..., 0..., 0, 0..., 0...]
        var k = projReshaped[0..., 0..., 1, 0..., 0...]
        let v = projReshaped[0..., 0..., 2, 0..., 0...]

        // Get offset from cache
        let offset = cache?.offset ?? 0

        // Apply RoPE if available
        if let rope = rope {
            (q, k) = rope(q, k, offset: offset)
        }

        // Transpose for attention: [B, T, H, D] -> [B, H, T, D]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        var vTransposed = v.transposed(0, 2, 1, 3)

        // Update cache and get full K, V
        let kFull: MLXArray
        let vFull: MLXArray

        if let cache = cache {
            (kFull, vFull) = cache.update(keys: k, values: vTransposed)
        } else {
            kFull = k
            vFull = vTransposed
        }

        // Create causal mask
        let mask = createAdditiveCausalMask(n: t, offset: offset).asType(query.dtype)

        // Scaled dot-product attention
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kFull,
            values: vFull,
            scale: scale,
            mask: mask
        )

        // Transpose back and reshape: [B, H, T, D] -> [B, T, H*D]
        let outTransposed = out.transposed(0, 2, 1, 3).reshaped([b, t, embedDim])

        return outProj(outTransposed)
    }
}

// MARK: - Simple KV Cache for Streaming

/// Simple KV cache implementation for streaming attention
public class StreamingKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0

    private let numHeads: Int
    private let headDim: Int
    private var step: Int = 256

    public init(numHeads: Int, headDim: Int, step: Int = 256) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.step = step
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }

    public func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let b = k.shape[0]
        let t = k.shape[2]  // [B, H, T, D]

        ensureCapacity(timeToAppend: t, batch: b, dtype: k.dtype)

        let prev = offset
        offset += t

        if var kBase = keys, var vBase = values {
            // Update keys and values at the right position
            keys = replaceSlice(base: kBase, axis: 2, start: prev, length: t, with: k)
            values = replaceSlice(base: vBase, axis: 2, start: prev, length: t, with: v)
        }

        // Return only the used portion
        let kUsed = MLX.split(keys!, indices: [offset], axis: 2)[0]
        let vUsed = MLX.split(values!, indices: [offset], axis: 2)[0]

        return (kUsed, vUsed)
    }

    private func ensureCapacity(timeToAppend t: Int, batch b: Int, dtype: DType) {
        let prev = offset
        if keys == nil || (prev + t) > keys!.shape[2] {
            let nSteps = (t + step - 1) / step
            let allocT = nSteps * step

            let newK = MLXArray.zeros([b, numHeads, allocT, headDim]).asType(dtype)
            let newV = MLXArray.zeros([b, numHeads, allocT, headDim]).asType(dtype)

            if var kExisting = keys, var vExisting = values {
                if prev % step != 0 {
                    kExisting = MLX.split(kExisting, indices: [prev], axis: 2)[0]
                    vExisting = MLX.split(vExisting, indices: [prev], axis: 2)[0]
                }
                keys = MLX.concatenated([kExisting, newK], axis: 2)
                values = MLX.concatenated([vExisting, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
    }

    private func replaceSlice(base: MLXArray, axis: Int, start: Int, length: Int, with repl: MLXArray) -> MLXArray {
        let split1 = MLX.split(base, indices: [start], axis: axis)
        let left = split1[0]
        let right = split1[1]
        let split2 = MLX.split(right, indices: [length], axis: axis)
        let rightRest = split2.count > 1 ? split2[1] : MLXArray.zeros([0])
        return MLX.concatenated([left, repl, rightRest], axis: axis)
    }
}
