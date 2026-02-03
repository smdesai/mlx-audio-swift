//
//  CausalMask.swift
//  Swift-TTS
//
//  Causal attention mask utilities for autoregressive generation.
//

import Foundation
import MLX
import MLXNN

// MARK: - Causal Mask Creation

/// Create an additive causal mask for attention
/// - Parameters:
///   - n: Sequence length
///   - offset: Position offset for streaming
/// - Returns: Mask tensor of shape [N, offset + N] with -inf for masked positions
public func createAdditiveCausalMask(n: Int, offset: Int = 0) -> MLXArray {
    // Row indices: offset to offset+N-1
    // Column indices: 0 to offset+N-1
    // Mask where col > row (future positions)

    let totalLen = offset + n
    let rowIndices = MLXArray(offset..<(offset + n))  // [N]
    let colIndices = MLXArray(0..<totalLen)  // [offset + N]

    // Create mask: positions where column > row should be masked
    // rowIndices[:, None] < colIndices[None, :] gives True where col > row
    let rowExpanded = rowIndices.reshaped([n, 1])
    let colExpanded = colIndices.reshaped([1, totalLen])

    let mask = MLX.less(rowExpanded, colExpanded)

    // Convert boolean mask to additive mask: True -> -1e9, False -> 0
    return MLX.where(mask, MLXArray(-1e9), MLXArray(0.0))
}

/// Create a causal mask for the full sequence (no offset)
public func createCausalMask(length: Int) -> MLXArray {
    return createAdditiveCausalMask(n: length, offset: 0)
}

// MARK: - Attention Mask Utilities

/// Combine causal mask with optional padding mask
/// - Parameters:
///   - causalMask: Causal attention mask
///   - paddingMask: Optional padding mask of shape [B, T] where True = padded
/// - Returns: Combined mask
public func combineMasks(
    causalMask: MLXArray,
    paddingMask: MLXArray?
) -> MLXArray {
    guard let padMask = paddingMask else {
        return causalMask
    }

    // Expand padding mask for broadcasting with attention
    // padding_mask: [B, T] -> [B, 1, 1, T]
    let expandedPadMask = padMask.reshaped([padMask.shape[0], 1, 1, padMask.shape[1]])

    // Convert to additive mask
    let paddingAdditive = MLX.where(expandedPadMask, MLXArray(-1e9), MLXArray(0.0))

    // Combine with causal mask
    return causalMask + paddingAdditive
}

// MARK: - Layer Scale

/// Layer scale module for transformer layers (PocketTTS-specific)
public class PocketCausalLayerScale: Module {
    private var scale: MLXArray

    public init(channels: Int, initValue: Float) {
        self.scale = MLXArray.full([channels], values: MLXArray(initValue))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * scale
    }
}
