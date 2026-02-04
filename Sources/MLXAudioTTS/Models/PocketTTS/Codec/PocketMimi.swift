//
//  PocketMimi.swift
//  Swift-TTS
//
//  Standalone Mimi implementation for PocketTTS with correct architecture:
//  - SEANet ratios: [6, 5, 4] (3 upsample/downsample layers)
//  - Transformer layers: 2
//
//  This avoids type conflicts with Marvis/Sesame Mimi implementations.
//

import Foundation
import MLX
import MLXNN

// MARK: - PocketTTS Mimi Configs

public struct PocketSeanetConfig {
    public let dimension: Int
    public let channels: Int
    public let causal: Bool
    public let nfilters: Int
    public let nresidualLayers: Int
    public let ratios: [Int]
    public let ksize: Int
    public let residualKsize: Int
    public let lastKsize: Int
    public let dilationBase: Int
    public let padMode: PadMode
    public let trueSkip: Bool
    public let compress: Int

    public init(
        dimension: Int = 512,
        channels: Int = 1,
        causal: Bool = true,
        nfilters: Int = 64,
        nresidualLayers: Int = 1,
        ratios: [Int] = [6, 5, 4],  // PocketTTS: 3 layers
        ksize: Int = 7,
        residualKsize: Int = 3,
        lastKsize: Int = 3,
        dilationBase: Int = 2,
        padMode: PadMode = .constant,
        trueSkip: Bool = true,
        compress: Int = 2
    ) {
        self.dimension = dimension
        self.channels = channels
        self.causal = causal
        self.nfilters = nfilters
        self.nresidualLayers = nresidualLayers
        self.ratios = ratios
        self.ksize = ksize
        self.residualKsize = residualKsize
        self.lastKsize = lastKsize
        self.dilationBase = dilationBase
        self.padMode = padMode
        self.trueSkip = trueSkip
        self.compress = compress
    }
}

public struct PocketTransformerConfig {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let causal: Bool
    public let normFirst: Bool
    public let biasFF: Bool
    public let biasAttn: Bool
    public let layerScale: Float?
    public let positionalEmbedding: String
    public let useConvBlock: Bool
    public let crossAttention: Bool
    public let convKernelSize: Int
    public let useConvBias: Bool
    public let gating: Bool
    public let norm: String
    public let context: Int
    public let maxPeriod: Int
    public let maxSeqLen: Int
    public let kvRepeat: Int
    public let dimFeedforward: Int
    public let convLayout: Bool

    public init(
        dModel: Int = 512,
        numHeads: Int = 8,
        numLayers: Int = 2,  // PocketTTS: 2 transformer layers
        causal: Bool = true,
        normFirst: Bool = true,
        biasFF: Bool = false,
        biasAttn: Bool = false,
        layerScale: Float? = 0.01,
        positionalEmbedding: String = "rope",
        useConvBlock: Bool = false,
        crossAttention: Bool = false,
        convKernelSize: Int = 3,
        useConvBias: Bool = true,
        gating: Bool = false,
        norm: String = "layer_norm",
        context: Int = 250,
        maxPeriod: Int = 10_000,
        maxSeqLen: Int = 8_192,
        kvRepeat: Int = 1,
        dimFeedforward: Int = 2_048,
        convLayout: Bool = true
    ) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.causal = causal
        self.normFirst = normFirst
        self.biasFF = biasFF
        self.biasAttn = biasAttn
        self.layerScale = layerScale
        self.positionalEmbedding = positionalEmbedding
        self.useConvBlock = useConvBlock
        self.crossAttention = crossAttention
        self.convKernelSize = convKernelSize
        self.useConvBias = useConvBias
        self.gating = gating
        self.norm = norm
        self.context = context
        self.maxPeriod = maxPeriod
        self.maxSeqLen = maxSeqLen
        self.kvRepeat = kvRepeat
        self.dimFeedforward = dimFeedforward
        self.convLayout = convLayout
    }

    public var headDim: Int { dModel / numHeads }
}

public struct PocketMimiConfig {
    public let channels: Int
    public let sampleRate: Double
    public let frameRate: Double
    public let renormalize: Bool
    public let seanet: PocketSeanetConfig
    public let transformer: PocketTransformerConfig
    public let quantizerNQ: Int
    public let quantizerBins: Int
    public let quantizerDim: Int

    public init(
        channels: Int = 1,
        sampleRate: Double = 24_000,
        frameRate: Double = 12.5,
        renormalize: Bool = true,
        seanet: PocketSeanetConfig = PocketSeanetConfig(),
        transformer: PocketTransformerConfig = PocketTransformerConfig(),
        quantizerNQ: Int = 32,
        quantizerBins: Int = 2_048,
        quantizerDim: Int = 256
    ) {
        self.channels = channels
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.renormalize = renormalize
        self.seanet = seanet
        self.transformer = transformer
        self.quantizerNQ = quantizerNQ
        self.quantizerBins = quantizerBins
        self.quantizerDim = quantizerDim
    }
}

/// Factory function for PocketTTS Mimi config
public func pocketMimiConfig(numCodebooks: Int = 32) -> PocketMimiConfig {
    let seanet = PocketSeanetConfig(
        dimension: 512,
        channels: 1,
        causal: true,
        nfilters: 64,
        nresidualLayers: 1,
        ratios: [6, 5, 4],  // PocketTTS architecture
        ksize: 7,
        residualKsize: 3,
        lastKsize: 3,
        dilationBase: 2,
        padMode: .constant,
        trueSkip: true,
        compress: 2
    )
    let transformer = PocketTransformerConfig(
        dModel: seanet.dimension,
        numHeads: 8,
        numLayers: 2,  // PocketTTS architecture
        causal: true,
        normFirst: true,
        biasFF: false,
        biasAttn: false,
        layerScale: 0.01,
        positionalEmbedding: "rope",
        useConvBlock: false,
        crossAttention: false,
        convKernelSize: 3,
        useConvBias: true,
        gating: false,
        norm: "layer_norm",
        context: 250,
        maxPeriod: 10_000,
        maxSeqLen: 8_192,
        kvRepeat: 1,
        dimFeedforward: 2_048,
        convLayout: true
    )
    return PocketMimiConfig(
        channels: 1,
        sampleRate: 24_000,
        frameRate: 12.5,
        renormalize: true,
        seanet: seanet,
        transformer: transformer,
        quantizerNQ: numCodebooks,
        quantizerBins: 2_048,
        quantizerDim: 256
    )
}

// MARK: - Helper Functions

@inline(__always) private func pocketProduct(_ xs: [Int]) -> Int { xs.reduce(1, *) }

// MARK: - PocketStreamingAdd

public final class PocketStreamingAdd: Module {
    private var lhsHold: MLXArray? = nil
    private var rhsHold: MLXArray? = nil

    override public init() {}

    public func step(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        var l = lhs
        var r = rhs

        if let h = lhsHold {
            l = concatenated([h, l], axis: 2)
            lhsHold = nil
        }
        if let h = rhsHold {
            r = concatenated([h, r], axis: 2)
            rhsHold = nil
        }

        let ll = l.shape[2]
        let rl = r.shape[2]

        if ll == rl {
            return l + r
        } else if ll < rl {
            let parts = split(r, indices: [ll], axis: 2)
            rhsHold = parts.count > 1 ? parts[1] : nil
            return l + parts[0]
        } else {
            let parts = split(l, indices: [rl], axis: 2)
            lhsHold = parts.count > 1 ? parts[1] : nil
            return parts[0] + r
        }
    }

    public func reset() {
        lhsHold = nil
        rhsHold = nil
    }
}

// MARK: - PocketSeanetResnetBlock

public final class PocketSeanetResnetBlock: Module {
    @ModuleInfo public var block: [PocketStreamableConv1d]
    @ModuleInfo(key: "streaming_add") public var streamingAdd = PocketStreamingAdd()
    @ModuleInfo public var shortcut: PocketStreamableConv1d?

    public init(cfg: PocketSeanetConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
        var layers: [PocketStreamableConv1d] = []
        let hidden = dim / cfg.compress
        for (i, kd) in ksizesAndDilations.enumerated() {
            let (ksize, dilation) = kd
            let inC = (i == 0) ? dim : hidden
            let outC = (i == ksizesAndDilations.count - 1) ? dim : hidden
            layers.append(PocketStreamableConv1d(
                inChannels: inC, outChannels: outC, ksize: ksize,
                stride: 1, dilation: dilation, groups: 1, bias: true,
                causal: cfg.causal, padMode: cfg.padMode
            ))
        }
        self._block = ModuleInfo(wrappedValue: layers)

        if cfg.trueSkip {
            self._shortcut = ModuleInfo(wrappedValue: nil)
        } else {
            self._shortcut = ModuleInfo(wrappedValue: PocketStreamableConv1d(
                inChannels: dim, outChannels: dim, ksize: 1,
                stride: 1, dilation: 1, groups: 1, bias: true,
                causal: cfg.causal, padMode: cfg.padMode
            ))
        }
    }

    public func resetState() {
        shortcut?.resetState()
        for b in block { b.resetState() }
        streamingAdd.reset()
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for b in block {
            x = b(elu(x, alpha: 1.0))
        }
        if let sc = shortcut {
            x = x + sc(xs)
        } else {
            x = x + xs
        }
        return x
    }

    public func step(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for b in block {
            x = b.step(elu(x, alpha: 1.0))
        }
        if let sc = shortcut {
            return streamingAdd.step(lhs: x, rhs: sc.step(xs))
        } else {
            return streamingAdd.step(lhs: x, rhs: xs)
        }
    }
}

// MARK: - PocketEncoderLayer

public final class PocketEncoderLayer: Module {
    @ModuleInfo public var residuals: [PocketSeanetResnetBlock]
    @ModuleInfo public var downsample: PocketStreamableConv1d

    public init(cfg: PocketSeanetConfig, ratio: Int, mult: Int) {
        var res: [PocketSeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<cfg.nresidualLayers {
            res.append(PocketSeanetResnetBlock(
                cfg: cfg,
                dim: mult * cfg.nfilters,
                ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]
            ))
            dilation *= cfg.dilationBase
        }
        self._residuals = ModuleInfo(wrappedValue: res)

        self._downsample = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: mult * cfg.nfilters,
            outChannels: mult * cfg.nfilters * 2,
            ksize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: cfg.padMode
        ))
    }

    public func resetState() {
        downsample.resetState()
        for r in residuals { r.resetState() }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for r in residuals { x = r(x) }
        return downsample(elu(x, alpha: 1.0))
    }

    public func step(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for r in residuals { x = r.step(x) }
        return downsample.step(elu(x, alpha: 1.0))
    }
}

// MARK: - PocketSeanetEncoder

public final class PocketSeanetEncoder: Module {
    @ModuleInfo public var init_conv1d: PocketStreamableConv1d
    @ModuleInfo public var layers: [PocketEncoderLayer]
    @ModuleInfo public var final_conv1d: PocketStreamableConv1d

    public init(cfg: PocketSeanetConfig) {
        var mult = 1

        self._init_conv1d = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: cfg.channels, outChannels: mult * cfg.nfilters,
            ksize: cfg.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
            causal: cfg.causal, padMode: cfg.padMode
        ))

        var encLayers: [PocketEncoderLayer] = []
        for ratio in cfg.ratios.reversed() {
            encLayers.append(PocketEncoderLayer(cfg: cfg, ratio: ratio, mult: mult))
            mult *= 2
        }
        self._layers = ModuleInfo(wrappedValue: encLayers)

        self._final_conv1d = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: mult * cfg.nfilters, outChannels: cfg.dimension,
            ksize: cfg.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
            causal: cfg.causal, padMode: cfg.padMode
        ))
    }

    public func resetState() {
        init_conv1d.resetState()
        final_conv1d.resetState()
        for l in layers { l.resetState() }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = init_conv1d(xs)
        for l in layers { x = l(x) }
        x = elu(x, alpha: 1.0)
        return final_conv1d(x)
    }

    public func step(_ xs: MLXArray) -> MLXArray {
        var x = init_conv1d.step(xs)
        for l in layers { x = l.step(x) }
        x = elu(x, alpha: 1.0)
        return final_conv1d.step(x)
    }
}

// MARK: - PocketDecoderLayer

public final class PocketDecoderLayer: Module {
    @ModuleInfo public var upsample: PocketStreamableConvTranspose1d
    @ModuleInfo public var residuals: [PocketSeanetResnetBlock]

    public init(cfg: PocketSeanetConfig, ratio: Int, mult: Int) {
        self._upsample = ModuleInfo(wrappedValue: PocketStreamableConvTranspose1d(
            inChannels: mult * cfg.nfilters,
            outChannels: mult * cfg.nfilters / 2,
            ksize: ratio * 2,
            stride: ratio,
            groups: 1,
            bias: true,
            causal: cfg.causal
        ))

        var res: [PocketSeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<cfg.nresidualLayers {
            res.append(PocketSeanetResnetBlock(
                cfg: cfg,
                dim: mult * cfg.nfilters / 2,
                ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]
            ))
            dilation *= cfg.dilationBase
        }
        self._residuals = ModuleInfo(wrappedValue: res)
    }

    public func resetState() {
        upsample.resetState()
        for r in residuals { r.resetState() }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = upsample(elu(xs, alpha: 1.0))
        for r in residuals { x = r(x) }
        return x
    }

    public func step(_ xs: MLXArray) -> MLXArray {
        var x = upsample.step(elu(xs, alpha: 1.0))
        for r in residuals { x = r.step(x) }
        return x
    }
}

// MARK: - PocketSeanetDecoder

public final class PocketSeanetDecoder: Module {
    @ModuleInfo public var init_conv1d: PocketStreamableConv1d
    @ModuleInfo public var layers: [PocketDecoderLayer]
    @ModuleInfo public var final_conv1d: PocketStreamableConv1d

    public init(cfg: PocketSeanetConfig) {
        var mult = 1 << cfg.ratios.count

        self._init_conv1d = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: cfg.dimension, outChannels: mult * cfg.nfilters,
            ksize: cfg.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
            causal: cfg.causal, padMode: cfg.padMode
        ))

        var decLayers: [PocketDecoderLayer] = []
        for ratio in cfg.ratios {
            decLayers.append(PocketDecoderLayer(cfg: cfg, ratio: ratio, mult: mult))
            mult /= 2
        }
        self._layers = ModuleInfo(wrappedValue: decLayers)

        self._final_conv1d = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: cfg.nfilters, outChannels: cfg.channels,
            ksize: cfg.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
            causal: cfg.causal, padMode: cfg.padMode
        ))
    }

    public func resetState() {
        init_conv1d.resetState()
        final_conv1d.resetState()
        for l in layers { l.resetState() }
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = init_conv1d(xs)
        for l in layers { x = l(x) }
        x = elu(x, alpha: 1.0)
        return final_conv1d(x)
    }

    public func step(_ xs: MLXArray) -> MLXArray {
        var x = init_conv1d.step(xs)
        for l in layers { x = l.step(x) }
        x = elu(x, alpha: 1.0)
        return final_conv1d.step(x)
    }
}

// MARK: - Convolution Components

@inline(__always)
fileprivate func pocketGetExtraPaddingForConv1d(xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let len = xs.shape[2]
    let nframes = max(len + paddingTotal - ksize, 0)
    let nf = Double(nframes) / Double(stride) + 1.0
    let idealLen = (Int(ceil(nf)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLen - len)
}

@inline(__always)
fileprivate func pocketUnpad1d(_ xs: MLXArray, unpadL: Int, unpadR: Int) -> MLXArray {
    let L = xs.shape[2]
    let parts = split(xs, indices: [unpadL, L - unpadR], axis: 2)
    return parts[1]
}

// MARK: - PocketConv1d

public final class PocketConv1d: Module {
    public var weight: MLXArray
    public var bias: MLXArray?

    public let padding: Int
    public let groups: Int
    public let stride: Int
    public let dilation: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        let scale: Float = 1.0 / Float(inChannels * ksize)
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, ksize, inChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        var y = conv1d(
            xsNLC, weight,
            stride: stride, padding: padding,
            dilation: dilation, groups: groups
        )
        if let b = bias { y = y + b }
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - PocketConvTranspose1d

public final class PocketConvTranspose1d: Module {
    public var weight: MLXArray
    public var bias: MLXArray?

    public let padding: Int
    public let groups: Int
    public let stride: Int
    public let ksize: Int
    public let inChannels: Int
    public let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale: Float = 1.0 / Float(inChannels * ksize)
        // Weight shape matches Python: (out_channels, ksize, in_channels // groups)
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, ksize, inChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.ksize = ksize
        self.inChannels = inChannels
        self.outChannels = outChannels
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        // MLX Swift's convTransposed1d supports groups natively - no expansion needed
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        var y = convTransposed1d(xsNLC, weight, stride: stride, padding: padding, groups: groups)
        if let b = bias { y = y + b }
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - PocketNormConv1d

public final class PocketNormConv1d: Module {
    @ModuleInfo public var conv: PocketConv1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, dilation: Int = 1, bias: Bool = true
    ) {
        self._conv = ModuleInfo(wrappedValue: PocketConv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, dilation: dilation, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
}

// MARK: - PocketNormConvTranspose1d

public final class PocketNormConvTranspose1d: Module {
    @ModuleInfo public var convtr: PocketConvTranspose1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, bias: Bool = true
    ) {
        self._convtr = ModuleInfo(wrappedValue: PocketConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
}

// MARK: - PocketStreamableConv1d

public final class PocketStreamableConv1d: Module {
    private let causal: Bool
    private let padMode: PadMode
    private let ksizeBase: Int
    @ModuleInfo public var conv: PocketNormConv1d

    private var prevXs: MLXArray? = nil
    private var leftPadApplied = false
    private let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool,
        causal: Bool,
        padMode: PadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self.ksizeBase = ksize
        self._conv = ModuleInfo(wrappedValue: PocketNormConv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, dilation: dilation, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() {
        prevXs = nil
        leftPadApplied = false
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let dil = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dil + 1
        let paddingTotal = kEff - conv.conv.stride
        let extra = pocketGetExtraPaddingForConv1d(
            xs: xsNCL, ksize: kEff, stride: conv.conv.stride, paddingTotal: paddingTotal
        )
        let z = IntOrPair(0)
        let pad: (Int, Int) = {
            if causal { return (paddingTotal, 0) }
            let pr = paddingTotal / 2
            return (paddingTotal - pr, pr)
        }()
        let (padL, padR) = pad
        let widths: [IntOrPair] = [z, z, IntOrPair((padL, padR + extra))]
        let xPad = padded(xsNCL, widths: widths, mode: padMode)
        return conv(xPad)
    }

    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = conv.conv.stride
        let dilation = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dilation + 1

        var x = xsNCL
        if !leftPadApplied {
            leftPadApplied = true
            let padTotal = kEff - stride
            x = padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((padTotal, 0))], mode: padMode)
        }

        if let prev = prevXs {
            x = concatenated([prev, x], axis: 2)
        }

        let L = x.shape[2]
        let nframes = max(L + stride - kEff, 0) / stride
        if nframes > 0 {
            let offset = nframes * stride
            let tailSplit = split(x, indices: [offset], axis: 2)
            prevXs = tailSplit.count > 1 ? tailSplit[1] : nil

            let inLen = (nframes - 1) * stride + kEff
            let keep = split(x, indices: [inLen], axis: 2)[0]
            return conv(keep)
        } else {
            prevXs = x
            return MLXArray.zeros([b, outChannels, 0])
        }
    }
}

// MARK: - PocketStreamableConvTranspose1d

public final class PocketStreamableConvTranspose1d: Module {
    private let causal: Bool
    private let ksize: Int
    @ModuleInfo public var convtr: PocketNormConvTranspose1d

    private var prevYs: MLXArray? = nil
    private let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        groups: Int,
        bias: Bool,
        causal: Bool
    ) {
        self.causal = causal
        self.ksize = ksize
        self._convtr = ModuleInfo(wrappedValue: PocketNormConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() { prevYs = nil }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let stride = convtr.convtr.stride
        let paddingTotal = max(ksize - stride, 0)
        let y = convtr(xsNCL)
        let (unL, unR): (Int, Int) = {
            if causal { return (0, paddingTotal) }
            let r = paddingTotal / 2
            return (paddingTotal - r, r)
        }()
        return pocketUnpad1d(y, unpadL: unL, unpadR: unR)
    }

    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = convtr.convtr.stride
        var y = convtr(xsNCL)
        let ot = y.shape[2]

        if var prev = prevYs {
            let pt = prev.shape[2]
            if let b = convtr.convtr.bias { prev = prev - b.reshaped([1, b.shape[0], 1]) }
            let head = split(y, indices: [pt], axis: 2)
            let combined = head[0] + prev
            y = concatenated([combined, head[1]], axis: 2)
        }

        let invalid = ksize - stride
        let parts = split(y, indices: [max(ot - invalid, 0)], axis: 2)
        let valid = parts[0]
        prevYs = parts.count > 1 ? parts[1] : nil
        return valid
    }
}

// MARK: - PocketConvDownsample1d

public final class PocketConvDownsample1d: Module {
    @ModuleInfo public var conv: PocketStreamableConv1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._conv = ModuleInfo(wrappedValue: PocketStreamableConv1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, dilation: 1, groups: 1, bias: false,
            causal: causal, padMode: .edge
        ))
    }

    public func resetState() { conv.resetState() }
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
    public func step(_ xs: MLXArray) -> MLXArray { conv.step(xs) }
}

// MARK: - PocketConvTrUpsample1d

public final class PocketConvTrUpsample1d: Module {
    @ModuleInfo public var convtr: PocketStreamableConvTranspose1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._convtr = ModuleInfo(wrappedValue: PocketStreamableConvTranspose1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, groups: dim, bias: false, causal: causal
        ))
    }

    public func resetState() { convtr.resetState() }
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
    public func step(_ xs: MLXArray) -> MLXArray { convtr.step(xs) }
}

// MARK: - Transformer Components

@inline(__always)
fileprivate func pocketGeluApprox(_ x: MLXArray) -> MLXArray {
    let c0 = MLXArray(0.7978845608028654)
    let c1 = MLXArray(0.044715)
    let x3 = x * x * x
    return 0.5 * x * (1 + tanh(c0 * (x + c1 * x3)))
}

public final class PocketId: Module {
    override public init() {}
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { xs }
}

public final class PocketLayerScale: Module {
    @ModuleInfo public var scale: MLXArray
    public init(dim: Int) {
        self._scale = ModuleInfo(wrappedValue: MLXArray.ones([dim]))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        xs * scale
    }
}

// MARK: - PocketKVCache

public final class PocketKVCache {
    public let nKVHeads: Int
    public let kHeadDim: Int
    public let vHeadDim: Int

    public private(set) var keys: MLXArray? = nil
    public private(set) var values: MLXArray? = nil
    public private(set) var offset: Int = 0
    public var step: Int = 256

    public init(headDim: Int, nKVHeads: Int, step: Int = 256) {
        self.nKVHeads = nKVHeads
        self.kHeadDim = headDim
        self.vHeadDim = headDim
        self.step = step
    }

    public func reset() {
        offset = 0
        keys = nil
        values = nil
    }

    public func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let B = k.shape[0]
        precondition(k.shape[1] == nKVHeads && v.shape[1] == nKVHeads, "nKVHeads mismatch")
        let t = k.shape[2]
        precondition(k.shape[3] == kHeadDim, "k head dim mismatch")
        precondition(v.shape[3] == vHeadDim, "v head dim mismatch")
        if let kk = keys { precondition(kk.shape[0] == B, "batch size changed in KV cache") }

        ensureCapacity(timeToAppend: t, batch: B, kDType: k.dtype, vDType: v.dtype)

        let prev = offset
        offset += t

        if let kBase = keys, let vBase = values {
            keys = replaceSlice(base: kBase, axis: 2, start: prev, length: t, with: k)
            values = replaceSlice(base: vBase, axis: 2, start: prev, length: t, with: v)
        }

        let kUsed = split(keys!, indices: [offset], axis: 2)[0]
        let vUsed = split(values!, indices: [offset], axis: 2)[0]
        return (kUsed, vUsed)
    }

    private func ensureCapacity(timeToAppend t: Int, batch B: Int, kDType: DType, vDType: DType) {
        let prev = offset
        if keys == nil || (prev + t) > keys!.shape[2] {
            let nSteps = (t + step - 1) / step
            let allocT = nSteps * step

            let newK = MLXArray.zeros([B, nKVHeads, allocT, kHeadDim]).asType(kDType)
            let newV = MLXArray.zeros([B, nKVHeads, allocT, vHeadDim]).asType(vDType)

            if var kExisting = keys, var vExisting = values {
                if prev % step != 0 {
                    kExisting = split(kExisting, indices: [prev], axis: 2)[0]
                    vExisting = split(vExisting, indices: [prev], axis: 2)[0]
                }
                keys = concatenated([kExisting, newK], axis: 2)
                values = concatenated([vExisting, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
    }

    private func replaceSlice(base: MLXArray, axis: Int, start: Int, length: Int, with repl: MLXArray) -> MLXArray {
        let split1 = split(base, indices: [start], axis: axis)
        let left = split1[0]
        let right = split1[1]
        let split2 = split(right, indices: [length], axis: axis)
        let rightRest = split2.count > 1 ? split2[1] : concatenated([], axis: axis)
        return concatenated([left, repl, rightRest], axis: axis)
    }
}

// MARK: - PocketAttention

public final class PocketAttention: Module {
    private let cfg: PocketTransformerConfig
    @ModuleInfo public var in_proj: Linear
    @ModuleInfo public var out_proj: Linear
    @ModuleInfo public var rope: RoPE?

    private let scale: Float

    public init(cfg: PocketTransformerConfig) {
        self.cfg = cfg
        precondition(cfg.kvRepeat == 1, "only kv_repeat == 1 is supported")

        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * (cfg.dModel / cfg.numHeads)
        self._in_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, outDim, bias: cfg.biasAttn))
        self._out_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())

        if cfg.positionalEmbedding == "rope" {
            self._rope = ModuleInfo(wrappedValue: RoPE(dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod)))
        } else {
            self._rope = ModuleInfo(wrappedValue: nil)
        }
    }

    public func callAsFunction(
        _ xs: MLXArray,
        cache: PocketKVCache,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]
        let hd = xs.shape[2]

        let qkv = in_proj(xs).reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        var q = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 0, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var k = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 1, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var v = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 2, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)

        if let rope {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        }

        (k, v) = cache.updateAndFetch(k, v)

        let kLen = k.shape[2]
        let kTargetLen = t + min(cfg.context, kLen - t)
        if kTargetLen < kLen {
            let start = kLen - kTargetLen
            k = split(k, indices: [start], axis: 2)[1]
            v = split(v, indices: [start], axis: 2)[1]
        }

        // Create causal attention mask with context window (matching Python implementation)
        // After slicing, k has currentKLen positions
        // q positions: [offset, offset+1, ..., offset+t-1]
        // k positions after slicing: [offset+t-currentKLen, ..., offset+t-1]
        let currentKLen = k.shape[2]
        let offset = cache.offset

        // Build position arrays for mask computation
        // pos_q shape: [t, 1], pos_k shape: [1, currentKLen]
        let posQ = MLXArray(Int32(offset)..<Int32(offset + t)).reshaped([t, 1])
        let kStartPos = offset + t - currentKLen
        let posK = MLXArray(Int32(kStartPos)..<Int32(kStartPos + currentKLen)).reshaped([1, currentKLen])

        // delta[i,j] = pos_q[i] - pos_k[j] = query position - key position
        let delta = posQ - posK  // [t, currentKLen]

        // Attention mask: (Python's attn_bias)
        // 1. pos_k >= 0: valid positions only
        // 2. delta >= 0: causal (query only attends to same or earlier positions)
        // 3. delta < context: within context window
        // Break up expression to help Swift compiler
        let validPos: MLXArray = posK .>= 0
        let causal: MLXArray = delta .>= 0
        let withinContext: MLXArray = delta .< Int32(cfg.context)
        let causalMask = validPos .&& causal .&& withinContext

        // Expand to [1, 1, t, currentKLen] for broadcasting with [b, h, t, currentKLen] scores
        let expandedMask = causalMask.reshaped([1, 1, t, currentKLen])

        var out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: expandedMask)
        out = swappedAxes(out, 1, 2).reshaped([b, t, hd])
        return out_proj(out)
    }
}

// MARK: - PocketMlpNoGating

public final class PocketMlpNoGating: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(cfg: PocketTransformerConfig) {
        self._linear1 = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dimFeedforward, bias: cfg.biasFF))
        self._linear2 = ModuleInfo(wrappedValue: Linear(cfg.dimFeedforward, cfg.dModel, bias: cfg.biasFF))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        linear2(pocketGeluApprox(linear1(xs)))
    }
}

// MARK: - PocketTransformerLayer

public final class PocketTransformerLayer: Module {
    @ModuleInfo public var gating: PocketMlpNoGating
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var norm2: LayerNorm
    @ModuleInfo public var layer_scale_1: Module
    @ModuleInfo public var layer_scale_2: Module
    @ModuleInfo public var self_attn: PocketAttention

    public init(cfg: PocketTransformerConfig) {
        precondition(!cfg.useConvBlock, "conv-block is not supported")
        precondition(!cfg.crossAttention, "cross-attn is not supported")

        self._gating = ModuleInfo(wrappedValue: PocketMlpNoGating(cfg: cfg))
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))

        if let _ = cfg.layerScale {
            self._layer_scale_1 = ModuleInfo(wrappedValue: PocketLayerScale(dim: cfg.dModel))
            self._layer_scale_2 = ModuleInfo(wrappedValue: PocketLayerScale(dim: cfg.dModel))
        } else {
            self._layer_scale_1 = ModuleInfo(wrappedValue: PocketId())
            self._layer_scale_2 = ModuleInfo(wrappedValue: PocketId())
        }

        self._self_attn = ModuleInfo(wrappedValue: PocketAttention(cfg: cfg))
    }

    public func callAsFunction(
        _ xs: MLXArray,
        cache: PocketKVCache
    ) -> MLXArray {
        var x = xs
        var n1 = norm1(x)

        n1 = self_attn(n1, cache: cache)

        if let ls = layer_scale_1 as? PocketLayerScale {
            x = x + ls(n1)
        } else {
            x = x + n1
        }
        let n2 = gating(norm2(x))
        if let ls = layer_scale_2 as? PocketLayerScale {
            x = x + ls(n2)
        } else {
            x = x + n2
        }
        return x
    }
}

// MARK: - PocketTransformer

public final class PocketTransformer: Module {
    private let cfg: PocketTransformerConfig
    @ModuleInfo public var layers: [PocketTransformerLayer]

    public init(cfg: PocketTransformerConfig) {
        self.cfg = cfg
        self._layers = ModuleInfo(wrappedValue: (0..<cfg.numLayers).map { _ in PocketTransformerLayer(cfg: cfg) })
    }

    public func callAsFunction(
        _ xs: MLXArray,
        cache: [PocketKVCache]
    ) -> MLXArray {
        var x = xs
        for (layer, c) in zip(layers, cache) {
            x = layer(x, cache: c)
        }
        return x
    }

    public func makeCache() -> [PocketKVCache] {
        let numKVHeads = cfg.numHeads / cfg.kvRepeat
        return (0..<cfg.numLayers).map { _ in PocketKVCache(headDim: cfg.headDim, nKVHeads: numKVHeads) }
    }
}

// MARK: - PocketProjectedTransformer

public final class PocketProjectedTransformer: Module {
    private let convLayout: Bool
    @ModuleInfo public var transformer: PocketTransformer
    @ModuleInfo public var input_proj: Linear?
    @ModuleInfo public var output_projs: [Linear?]

    public init(cfg: PocketTransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer = ModuleInfo(wrappedValue: PocketTransformer(cfg: cfg))

        if inputDim == cfg.dModel {
            self._input_proj = ModuleInfo(wrappedValue: nil)
        } else {
            self._input_proj = ModuleInfo(wrappedValue: Linear(inputDim, cfg.dModel, bias: false))
        }

        var outs: [Linear?] = []
        for od in outputDims {
            if od == cfg.dModel {
                outs.append(nil)
            } else {
                outs.append(Linear(cfg.dModel, od, bias: false))
            }
        }
        self._output_projs = ModuleInfo(wrappedValue: outs)
    }

    public func callAsFunction(
        _ xsIn: MLXArray,
        cache: [PocketKVCache]
    ) -> [MLXArray] {
        var xs = xsIn
        if convLayout { xs = swappedAxes(xs, 1, 2) }

        if let ip = input_proj { xs = ip(xs) }

        xs = transformer(xs, cache: cache)

        if output_projs.compactMap({ $0 }).count == 0 {
            return [swappedAxes(xs, 1, 2)]
        } else {
            var outs: [MLXArray] = []
            for op in output_projs {
                guard let op else { continue }
                var out = op(xs)
                if convLayout { out = swappedAxes(out, 1, 2) }
                outs.append(out)
            }
            return outs
        }
    }

    public func makeCache() -> [PocketKVCache] { transformer.makeCache() }
}

// MARK: - PocketMimi

public final class PocketMimi: Module, @unchecked Sendable {
    public let cfg: PocketMimiConfig

    @ModuleInfo public var encoder: PocketSeanetEncoder
    @ModuleInfo public var decoder: PocketSeanetDecoder

    @ModuleInfo public var encoder_transformer: PocketProjectedTransformer
    @ModuleInfo public var decoder_transformer: PocketProjectedTransformer

    @ModuleInfo public var downsample: PocketConvDownsample1d
    @ModuleInfo public var upsample: PocketConvTrUpsample1d

    public private(set) var encoderCache: [PocketKVCache]
    public private(set) var decoderCache: [PocketKVCache]

    private let downsampleStride: Int

    public init(cfg: PocketMimiConfig) {
        self.cfg = cfg

        let encFPS = cfg.sampleRate / Double(pocketProduct(cfg.seanet.ratios))
        self.downsampleStride = Int(encFPS / cfg.frameRate)

        self._encoder = ModuleInfo(wrappedValue: PocketSeanetEncoder(cfg: cfg.seanet))
        self._decoder = ModuleInfo(wrappedValue: PocketSeanetDecoder(cfg: cfg.seanet))

        self._encoder_transformer = ModuleInfo(wrappedValue: PocketProjectedTransformer(
            cfg: cfg.transformer,
            inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]
        ))
        self._decoder_transformer = ModuleInfo(wrappedValue: PocketProjectedTransformer(
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

        self.encoderCache = _encoder_transformer.wrappedValue.makeCache()
        self.decoderCache = _decoder_transformer.wrappedValue.makeCache()
    }

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
        for c in decoderCache { c.reset() }
        for c in encoderCache { c.reset() }
    }

    public var frameRate: Double { cfg.frameRate }
    public var sampleRate: Double { cfg.sampleRate }

    // MARK: - Encoding (Audio -> Latent)

    public func encodeToLatent(_ xs: MLXArray) -> MLXArray {
        encoder.resetState()
        for c in encoderCache { c.reset() }

        var z = encoder(xs)
        z = encoder_transformer(z, cache: encoderCache)[0]
        z = downsample(z)
        return z
    }

    // MARK: - Decoding (Latent -> Audio)

    public func decodeFromLatent(_ latent: MLXArray) -> MLXArray {
        decoder.resetState()
        for c in decoderCache { c.reset() }
        upsample.resetState()

        var z = upsample(latent)
        z = decoder_transformer(z, cache: decoderCache)[0]
        return decoder(z)
    }

    public func decodeStep(_ latent: MLXArray) -> MLXArray {
        var z = upsample.step(latent)
        z = decoder_transformer(z, cache: decoderCache)[0]
        return decoder.step(z)
    }
}

// MARK: - PocketMimi Streaming Decoder

public final class PocketMimiStreamingDecoder {
    private let mimi: PocketMimi

    public init(_ mimi: PocketMimi) {
        self.mimi = mimi
        reset()
    }

    public func reset() {
        mimi.decoder.resetState()
        mimi.upsample.resetState()
        for c in mimi.decoderCache { c.reset() }
    }

    public func decodeFrames(_ latents: MLXArray) -> MLXArray {
        let lat = (latents.ndim == 2) ? latents.expandedDimensions(axes: [0]) : latents
        let T = lat.shape[2]

        var pcs: [MLXArray] = []
        for t in 0..<T {
            let left = split(lat, indices: [t], axis: 2)
            let mid = split(left[1], indices: [1], axis: 2)[0]
            pcs.append(mimi.decodeStep(mid))
        }
        return concatenated(pcs, axis: 2)
    }
}
