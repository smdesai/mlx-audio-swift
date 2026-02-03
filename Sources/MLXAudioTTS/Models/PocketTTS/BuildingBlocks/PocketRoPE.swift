//
//  PocketRoPE.swift
//  Swift-TTS
//
//  Rotary Position Embeddings for PocketTTS streaming transformer.
//

import Foundation
import MLX
import MLXNN

// MARK: - Apply RoPE Function

/// Apply rotary position embeddings to query and key tensors
/// - Parameters:
///   - q: Query tensor of shape [B, T, H, D]
///   - k: Key tensor of shape [B, T, H, D]
///   - offset: Position offset for streaming
///   - maxPeriod: Maximum period for frequency computation (default 10000)
/// - Returns: Tuple of (rotated_q, rotated_k) with same shapes
public func applyRoPE(
    q: MLXArray,
    k: MLXArray,
    offset: Int = 0,
    maxPeriod: Float = 10000.0
) -> (MLXArray, MLXArray) {
    let shape = q.shape
    let b = shape[0]
    let t = shape[1]
    let h = shape[2]
    let d = shape[3]

    precondition(d % 2 == 0, "RoPE requires an even head dimension.")
    let half = d / 2

    // Compute frequencies: exp(-log(maxPeriod) * 2 * i / d) for i in 0..<half
    let exponents = MLXArray(0..<half).asType(.float32)
    let logMaxPeriod = Float(log(Double(maxPeriod)))
    let freqScale: Float = -logMaxPeriod * 2.0 / Float(d)
    let freqs = MLX.exp(exponents * freqScale)

    // Time steps with offset
    let ts = (MLXArray(0..<t).asType(.float32) + Float(offset))
        .reshaped([1, t, 1, 1])  // [1, T, 1, 1]

    // Reshape q and k to separate real and imaginary parts
    let qReshaped = q.reshaped([b, t, h, half, 2])
    let kReshaped = k.reshaped([b, t, h, half, 2])

    // Extract real and imaginary components
    let qr = qReshaped[0..., 0..., 0..., 0..., 0].asType(.float32)
    let qi = qReshaped[0..., 0..., 0..., 0..., 1].asType(.float32)
    let kr = kReshaped[0..., 0..., 0..., 0..., 0].asType(.float32)
    let ki = kReshaped[0..., 0..., 0..., 0..., 1].asType(.float32)

    // Compute rotation matrix elements
    let freqsReshaped = freqs.reshaped([1, 1, 1, half])  // [1, 1, 1, half]
    let angles = freqsReshaped * ts  // [1, T, 1, half]
    let rotr = MLX.cos(angles)
    let roti = MLX.sin(angles)

    // Apply rotation: (qr + i*qi) * (rotr + i*roti)
    // Real: qr*rotr - qi*roti
    // Imag: qr*roti + qi*rotr
    let qor = qr * rotr - qi * roti
    let qoi = qr * roti + qi * rotr
    let kor = kr * rotr - ki * roti
    let koi = kr * roti + ki * rotr

    // Stack and reshape back
    let dtype = q.dtype
    let qOut = MLX.stacked([qor.asType(dtype), qoi.asType(dtype)], axis: -1)
        .reshaped([b, t, h, d])
    let kOut = MLX.stacked([kor.asType(dtype), koi.asType(dtype)], axis: -1)
        .reshaped([b, t, h, d])

    return (qOut, kOut)
}

// MARK: - Rotary Embedding Module

/// Rotary Embedding module for PocketTTS
public class RotaryEmbedding: Module {
    public let maxPeriod: Float

    public init(maxPeriod: Float = 10000.0) {
        self.maxPeriod = maxPeriod
        super.init()
    }

    /// Apply rotary embeddings to query and key tensors
    /// - Parameters:
    ///   - q: Query tensor of shape [B, T, H, D]
    ///   - k: Key tensor of shape [B, T, H, D]
    ///   - offset: Position offset for streaming/caching
    /// - Returns: Tuple of rotated (q, k)
    public func callAsFunction(_ q: MLXArray, _ k: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        return applyRoPE(q: q, k: k, offset: offset, maxPeriod: maxPeriod)
    }
}
