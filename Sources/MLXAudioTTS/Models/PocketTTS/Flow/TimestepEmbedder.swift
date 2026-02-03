//
//  TimestepEmbedder.swift
//  Swift-TTS
//
//  Timestep embedding for flow matching diffusion in PocketTTS.
//

import Foundation
import MLX
import MLXNN

// MARK: - RMS Normalization

/// RMS Normalization (used in flow network) - named FlowRMSNorm to avoid conflict with MLXNN.RMSNorm
public class FlowRMSNorm: Module, UnaryLayer {
    @ModuleInfo public var alpha: MLXArray
    private let eps: Float

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._alpha = ModuleInfo(wrappedValue: MLXArray.ones([dim]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xDtype = x.dtype
        let xFloat = x.asType(.float32)

        // Compute variance with ddof=1 (unbiased)
        let variance = MLX.variance(xFloat, axis: -1, keepDims: true, ddof: 1)
        let normalized = xFloat * (alpha.asType(.float32) * MLX.rsqrt(variance + eps))

        return normalized.asType(xDtype)
    }
}

// MARK: - Flow Layer Norm

/// Layer Norm that matches PyTorch reference for flow network
public class FlowLayerNorm: Module {
    private var weight: MLXArray?
    private var bias: MLXArray?
    private let eps: Float
    private let elementwiseAffine: Bool

    public init(channels: Int, eps: Float = 1e-6, elementwiseAffine: Bool = true) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine

        if elementwiseAffine {
            self.weight = MLXArray.ones([channels])
            self.bias = MLXArray.zeros([channels])
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

// MARK: - Timestep Embedder

/// Embeds scalar timesteps into vector representations using sinusoidal embeddings
public class TimestepEmbedder: Module {
    private let frequencyEmbeddingSize: Int
    private let freqs: MLXArray

    // MLP structure matches Python: Linear[0] -> SiLU[1] -> Linear[2] -> FlowRMSNorm[3]
    // We use Sequential to match weight key paths exactly
    @ModuleInfo public var mlp: Sequential

    public init(
        hiddenSize: Int,
        frequencyEmbeddingSize: Int = 256,
        maxPeriod: Int = 10000
    ) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize

        // Compute frequencies for sinusoidal embedding
        let half = frequencyEmbeddingSize / 2
        let exponents = MLXArray(0..<half).asType(.float32) / Float(half)
        self.freqs = MLX.exp(-Float(log(Double(maxPeriod))) * exponents)

        // MLP: Linear[0] -> SiLU[1] -> Linear[2] -> FlowRMSNorm[3]
        // Must match Python indices exactly for weight loading
        self._mlp = ModuleInfo(wrappedValue: Sequential(layers: [
            Linear(frequencyEmbeddingSize, hiddenSize, bias: true),  // index 0
            SiLU(),                                                  // index 1 (no weights)
            Linear(hiddenSize, hiddenSize, bias: true),              // index 2
            FlowRMSNorm(dim: hiddenSize)                             // index 3
        ]))

        super.init()
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        var timestep = t
        if timestep.ndim == 1 {
            timestep = timestep.reshaped([timestep.shape[0], 1])
        }

        // Compute sinusoidal embedding
        let args = timestep.asType(.float32) * freqs.reshaped([1, freqs.shape[0]])
        let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)

        // Pass through MLP (Sequential handles all layers)
        return mlp(embedding)
    }
}

// MARK: - Modulation Function

/// Modulate input with shift and scale from conditioning
/// - Parameters:
///   - x: Input tensor
///   - shift: Shift values
///   - scale: Scale values
/// - Returns: Modulated tensor: x * (1 + scale) + shift
public func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    return x * (1 + scale) + shift
}
