//
//  FlowMLP.swift
//  Swift-TTS
//
//  Simple MLP with AdaLN conditioning for flow matching in PocketTTS.
//

import Foundation
import MLX
import MLXNN

// MARK: - Simple MLP with AdaLN

/// Simple MLP with Adaptive Layer Normalization for flow matching
/// This is the main flow network that predicts velocity fields
public class SimpleMLPAdaLN: Module {
    public let inChannels: Int
    public let modelChannels: Int
    public let outChannels: Int
    public let numResBlocks: Int
    public let numTimeConds: Int

    private let timeEmbed: [TimestepEmbedder]
    private let condEmbed: Linear
    private let inputProj: Linear
    private let resBlocks: [ResBlock]
    private let finalLayer: FinalLayer

    public init(
        inChannels: Int,
        modelChannels: Int,
        outChannels: Int,
        condChannels: Int,
        numResBlocks: Int,
        numTimeConds: Int = 2
    ) {
        precondition(numTimeConds != 1, "numTimeConds must be != 1 for AdaLN conditioning.")

        self.inChannels = inChannels
        self.modelChannels = modelChannels
        self.outChannels = outChannels
        self.numResBlocks = numResBlocks
        self.numTimeConds = numTimeConds

        // Multiple timestep embedders (for s and t in LSD decode)
        self.timeEmbed = (0..<numTimeConds).map { _ in
            TimestepEmbedder(hiddenSize: modelChannels)
        }

        // Conditioning embedding
        self.condEmbed = Linear(condChannels, modelChannels, bias: true)

        // Input projection
        self.inputProj = Linear(inChannels, modelChannels, bias: true)

        // Residual blocks
        self.resBlocks = (0..<numResBlocks).map { _ in
            ResBlock(channels: modelChannels)
        }

        // Final output layer
        self.finalLayer = FinalLayer(modelChannels: modelChannels, outChannels: outChannels)

        super.init()
    }

    /// Forward pass
    /// - Parameters:
    ///   - c: Conditioning from transformer backbone [B, D]
    ///   - s: Start timestep [B] or [B, 1]
    ///   - t: End timestep [B] or [B, 1]
    ///   - x: Noisy latent input [B, T, latentDim]
    /// - Returns: Predicted velocity [B, T, outChannels]
    public func callAsFunction(
        c: MLXArray,
        s: MLXArray,
        t: MLXArray,
        x: MLXArray
    ) -> MLXArray {
        let ts = [s, t]
        precondition(ts.count == numTimeConds, "Expected \(numTimeConds) time conditions, got \(ts.count)")

        // Project input
        var out = inputProj(x)

        // Combine timestep embeddings
        var tCombined = MLXArray.zeros([out.shape[0], modelChannels]).asType(out.dtype)
        for timeEmbedder in timeEmbed.enumerated() {
            let tEmb = timeEmbedder.element(ts[timeEmbedder.offset])
            tCombined = tCombined + tEmb
        }
        tCombined = tCombined / Float(numTimeConds)

        // Add conditioning embedding
        let cEmbed = condEmbed(c)
        let y = tCombined + cEmbed

        // Pass through residual blocks
        for block in resBlocks {
            out = block(out, y: y)
        }

        // Final layer
        return finalLayer(out, c: y)
    }

    /// Create from configuration
    public static func fromConfig(
        _ config: FlowLMConfig,
        latentDim: Int,
        condDim: Int
    ) -> SimpleMLPAdaLN {
        return SimpleMLPAdaLN(
            inChannels: latentDim,
            modelChannels: config.flow.dim,
            outChannels: latentDim,
            condChannels: condDim,
            numResBlocks: config.flow.depth,
            numTimeConds: 2
        )
    }
}

// MARK: - LSD Decode (Low-Step Decoder)

/// Flow Type Definition
public typealias FlowNet = (MLXArray, MLXArray, MLXArray) -> MLXArray

/// LSD Decode algorithm for flow matching inference
/// - Parameters:
///   - vt: Flow network function (s, t, x) -> velocity
///   - x0: Initial noise [B, T, D]
///   - numSteps: Number of integration steps (default 1)
/// - Returns: Decoded latent
public func lsdDecode(
    vt: FlowNet,
    x0: MLXArray,
    numSteps: Int = 1
) -> MLXArray {
    var current = x0

    for i in 0..<numSteps {
        let s = Float(i) / Float(numSteps)
        let t = Float(i + 1) / Float(numSteps)

        // Create timestep tensors
        let shape = x0[0..., 0..., 0..<1].shape  // [B, T, 1]
        let sT = MLXArray.full(shape, values: MLXArray(s)).asType(x0.dtype)
        let tT = MLXArray.full(shape, values: MLXArray(t)).asType(x0.dtype)

        // Get flow direction
        let flowDir = vt(sT, tT, current)

        // Euler integration step
        current = current + flowDir / Float(numSteps)
    }

    return current
}

/// LSD Decode with partial application for conditioning
public func lsdDecodeConditioned(
    flowNet: SimpleMLPAdaLN,
    conditioning: MLXArray,
    x0: MLXArray,
    numSteps: Int = 1
) -> MLXArray {
    // Create flow function with conditioning partially applied
    let vt: FlowNet = { s, t, x in
        // Squeeze s and t to [B] for timestep embedder
        let sSqueezed = s.mean(axes: [1, 2])  // [B]
        let tSqueezed = t.mean(axes: [1, 2])  // [B]
        return flowNet(c: conditioning, s: sSqueezed, t: tSqueezed, x: x)
    }

    return lsdDecode(vt: vt, x0: x0, numSteps: numSteps)
}
