//
//  ResBlock.swift
//  Swift-TTS
//
//  Residual block with AdaLN modulation for PocketTTS flow network.
//

import Foundation
import MLX
import MLXNN

// MARK: - ResBlock with AdaLN Modulation

/// Residual block with Adaptive Layer Normalization for flow network
public class ResBlock: Module {
    private let inLN: FlowLayerNorm
    private let linear1: Linear
    private let linear2: Linear
    private let adaLNModulation: Sequential  // Sequential([SiLU, Linear]) to match Python
    private let channels: Int

    public init(channels: Int) {
        self.channels = channels
        self.inLN = FlowLayerNorm(channels: channels, eps: 1e-6)

        // MLP layers
        self.linear1 = Linear(channels, channels, bias: true)
        self.linear2 = Linear(channels, channels, bias: true)

        // AdaLN modulation: SiLU + Linear to 3*channels (shift, scale, gate)
        // Use Sequential to match Python model structure for weight loading
        self.adaLNModulation = Sequential(layers: [
            SiLU(),
            Linear(channels, 3 * channels, bias: true)
        ])

        super.init()
    }

    /// Forward pass with conditioning
    /// - Parameters:
    ///   - x: Input tensor [B, T, C]
    ///   - y: Conditioning tensor [B, C]
    /// - Returns: Output tensor [B, T, C]
    public func callAsFunction(_ x: MLXArray, y: MLXArray) -> MLXArray {
        // Compute modulation parameters (SiLU is now in Sequential)
        let modulation = adaLNModulation(y)
        let splitMod = MLX.split(modulation, parts: 3, axis: -1)
        let shiftMLP = splitMod[0]
        let scaleMLP = splitMod[1]
        let gateMLP = splitMod[2]

        // Expand conditioning for sequence dimension if needed
        let shiftExpanded = expandForSequence(shiftMLP, seqLen: x.shape[1])
        let scaleExpanded = expandForSequence(scaleMLP, seqLen: x.shape[1])
        let gateExpanded = expandForSequence(gateMLP, seqLen: x.shape[1])

        // Apply modulated layer norm
        var h = modulate(inLN(x), shift: shiftExpanded, scale: scaleExpanded)

        // MLP with SiLU
        h = linear1(h)
        h = silu(h)
        h = linear2(h)

        // Residual with gating
        return x + gateExpanded * h
    }

    /// Expand conditioning tensor for sequence dimension
    private func expandForSequence(_ cond: MLXArray, seqLen: Int) -> MLXArray {
        if cond.ndim == 2 {
            // [B, C] -> [B, 1, C] -> broadcast to [B, T, C]
            return cond.reshaped([cond.shape[0], 1, cond.shape[1]])
        }
        return cond
    }
}

// MARK: - Final Layer with AdaLN

/// Final layer with AdaLN modulation for flow network output
public class FinalLayer: Module {
    private let normFinal: FlowLayerNorm
    private let linear: Linear
    private let adaLNModulation: Sequential  // Sequential([SiLU, Linear]) to match Python
    private let modelChannels: Int

    public init(modelChannels: Int, outChannels: Int) {
        self.modelChannels = modelChannels

        // Layer norm without affine parameters
        self.normFinal = FlowLayerNorm(channels: modelChannels, eps: 1e-6, elementwiseAffine: false)

        // Output projection
        self.linear = Linear(modelChannels, outChannels, bias: true)

        // AdaLN modulation to 2*channels (shift, scale)
        // Use Sequential to match Python model structure for weight loading
        self.adaLNModulation = Sequential(layers: [
            SiLU(),
            Linear(modelChannels, 2 * modelChannels, bias: true)
        ])

        super.init()
    }

    /// Forward pass with conditioning
    /// - Parameters:
    ///   - x: Input tensor [B, T, C]
    ///   - c: Conditioning tensor [B, C]
    /// - Returns: Output tensor [B, T, outChannels]
    public func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        // Compute modulation parameters (SiLU is now in Sequential)
        let modulation = adaLNModulation(c)
        let splitMod = MLX.split(modulation, parts: 2, axis: -1)
        let shift = splitMod[0]
        let scale = splitMod[1]

        // Expand for sequence dimension
        let shiftExpanded = expandForSequence(shift, seqLen: x.shape[1])
        let scaleExpanded = expandForSequence(scale, seqLen: x.shape[1])

        // Apply modulated norm and project
        let normalized = modulate(normFinal(x), shift: shiftExpanded, scale: scaleExpanded)
        return linear(normalized)
    }

    private func expandForSequence(_ cond: MLXArray, seqLen: Int) -> MLXArray {
        if cond.ndim == 2 {
            return cond.reshaped([cond.shape[0], 1, cond.shape[1]])
        }
        return cond
    }
}
