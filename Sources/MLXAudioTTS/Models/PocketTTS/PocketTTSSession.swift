//
//  PocketTTSSession.swift
//  Swift-TTS
//
//  Observable session wrapper for PocketTTS with async streaming support.
//

import Foundation
import MLX
import MLXNN
@preconcurrency import AVFoundation

// MARK: - Generation Status

public enum PocketTTSGenerationStatus {
    case idle
    case loading
    case generating
    case streaming
    case completed
    case error(String)
}

// MARK: - PocketTTS Session

/// Observable session for PocketTTS generation
@MainActor
public class PocketTTSSession: ObservableObject {
    // MARK: - Published Properties

    @Published public private(set) var status: PocketTTSGenerationStatus = .idle
    @Published public private(set) var progress: Float = 0.0
    @Published public private(set) var generatedSamples: Int = 0
    @Published public private(set) var currentVoice: String = PocketTTSDefaults.defaultVoice

    // MARK: - Model

    private var model: PocketTTSModel?
    private var state: PocketTTSState?

    // MARK: - Audio Playback

    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioFormat: AVAudioFormat?

    // MARK: - Configuration

    public var temperature: Float {
        get { model?.temp ?? PocketTTSDefaults.temperature }
        set { model?.temp = newValue }
    }

    public var lsdDecodeSteps: Int {
        get { model?.lsdDecodeSteps ?? PocketTTSDefaults.lsdDecodeSteps }
        set { model?.lsdDecodeSteps = newValue }
    }

    public var sampleRate: Double {
        return Double(model?.sampleRate ?? 24000)
    }

    // MARK: - Initialization

    public init() {
        setupAudioEngine()
    }

    // MARK: - Model Loading

    /// Load PocketTTS model from configuration, creating PocketMimi internally
    /// Downloads weights from HuggingFace Hub
    public func loadModel(
        config: PocketTTSModelConfig,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws {
        guard let mimiConfig = config.mimi else {
            throw PocketTTSError.configurationMissing("mimi configuration required")
        }

        // Create PocketMimi from config
        let pocketMimiConfig = pocketMimiConfigFromJSON(mimiConfig)
        let mimi = PocketMimi(cfg: pocketMimiConfig)

        try await loadModel(config: config, mimi: mimi, repoId: repoId, progressHandler: progressHandler)
    }

    /// Load PocketTTS model from configuration with pre-created PocketMimi
    /// Downloads weights from HuggingFace Hub
    public func loadModel(
        config: PocketTTSModelConfig,
        mimi: PocketMimi,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws {
        status = .loading
        progress = 0.0

        do {
            model = try await PocketTTSModel.fromConfig(config, mimi: mimi, repoId: repoId)

            // Download and load weights from Hub
            guard let loadedModel = model else {
                throw PocketTTSError.configurationMissing("Model not initialized")
            }

            try await loadPocketTTSWeightsFromHub(
                model: loadedModel,
                repoId: repoId,
                filename: PocketTTSRepoInfo.defaultWeightsFile,
                progressHandler: { @Sendable hubProgress in
                    let fraction = Float(hubProgress.fractionCompleted)
                    Task { @MainActor [weak self] in
                        self?.progress = fraction
                    }
                }
            )
            progressHandler?(1.0)

            status = .idle
            progress = 1.0
        } catch {
            status = .error(error.localizedDescription)
            throw error
        }
    }

    /// Load model weights from path
    public func loadWeights(from path: String, progressHandler: ((Float) -> Void)? = nil) async throws {
        guard let model = model else {
            throw PocketTTSError.configurationMissing("Model not initialized")
        }

        let url = URL(fileURLWithPath: path)
        let weights = try MLX.loadArrays(url: url)

        // Apply weights to model
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)
    }

    // MARK: - Voice Selection

    /// Set voice by name (predefined voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma)
    public func setVoice(_ voiceName: String) async throws {
        guard let model = model else {
            throw PocketTTSError.configurationMissing("Model not loaded")
        }

        currentVoice = voiceName
        state = try await model.getStateForVoice(voiceName)
    }

    /// Set voice from audio file for voice cloning
    /// - Parameters:
    ///   - audioURL: URL to audio file (WAV, MP3, FLAC supported via AVFoundation)
    ///   - truncate: Whether to truncate to 30 seconds max (default: true)
    public func setVoiceFromAudio(_ audioURL: URL, truncate: Bool = true) async throws {
        guard let model = model else {
            throw PocketTTSError.configurationMissing("Model not loaded")
        }

        currentVoice = "custom"
        state = try model.getStateForAudioFile(audioURL, truncate: truncate)
    }

    // MARK: - Generation

    /// Generate speech from text (blocking)
    public func generate(text: String) async throws -> [Float] {
        guard var model = model else {
            throw PocketTTSError.configurationMissing("Model not loaded")
        }

        // Initialize state if needed
        if state == nil {
            state = try await model.getStateForVoice(currentVoice)
        }

        guard var currentState = state else {
            throw PocketTTSError.generationError("Failed to initialize generation state")
        }

        status = .generating
        generatedSamples = 0

        let audio = model.generateAudio(state: currentState, text: text)
        let samples = audio.asArray(Float.self)

        state = currentState
        generatedSamples = samples.count
        status = .completed

        return samples
    }

    /// Generate speech with streaming playback
    public func generateAndPlay(text: String) async throws {
        guard let model = model else {
            throw PocketTTSError.configurationMissing("Model not loaded")
        }

        // Initialize state if needed
        if state == nil {
            state = try await model.getStateForVoice(currentVoice)
        }

        guard let currentState = state else {
            throw PocketTTSError.generationError("Failed to initialize generation state")
        }

        status = .streaming
        generatedSamples = 0

        // Start playback
        startPlayback()

        // Stream audio chunks
        for chunk in model.generateAudioStream(state: currentState, text: text) {
            let samples = chunk.asArray(Float.self)
            generatedSamples += samples.count

            // Play chunk
            playAudioChunk(samples)

            // Allow UI updates
            await Task.yield()
        }

        state = currentState
        status = .completed
    }

    /// Generate speech as AsyncStream
    public func generateStream(text: String) -> AsyncStream<[Float]> {
        AsyncStream { continuation in
            Task {
                guard var model = self.model else {
                    continuation.finish()
                    return
                }

                // Initialize state if needed
                if self.state == nil {
                    do {
                        self.state = try await model.getStateForVoice(self.currentVoice)
                    } catch {
                        continuation.finish()
                        return
                    }
                }

                guard var currentState = self.state else {
                    continuation.finish()
                    return
                }

                await MainActor.run {
                    self.status = .streaming
                    self.generatedSamples = 0
                }

                for chunk in model.generateAudioStream(state: currentState, text: text) {
                    let samples = chunk.asArray(Float.self)

                    await MainActor.run {
                        self.generatedSamples += samples.count
                    }

                    continuation.yield(samples)
                }

                await MainActor.run {
                    self.state = currentState
                    self.status = .completed
                }

                continuation.finish()
            }
        }
    }

    // MARK: - Audio Engine

    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        guard let engine = audioEngine, let player = playerNode else { return }

        let rate = sampleRate
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: audioFormat)

        do {
            try engine.start()
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    private func startPlayback() {
        guard let player = playerNode else { return }
        if !player.isPlaying {
            player.play()
        }
    }

    private func playAudioChunk(_ samples: [Float]) {
        guard let format = audioFormat, let player = playerNode else { return }

        let frameLength = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength) else { return }

        buffer.frameLength = frameLength
        if let channelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                channelData[0][i] = samples[i]
            }
        }

        player.scheduleBuffer(buffer, completionHandler: nil)
    }

    public func stopPlayback() {
        playerNode?.stop()
    }

    // MARK: - Cleanup

    /// Clean up audio resources (call before discarding session)
    public func cleanup() {
        audioEngine?.stop()
        playerNode?.stop()
    }
}

// MARK: - Config Conversion Helpers

/// Convert JSON-parsed MimiConfig to PocketMimiConfig
/// Uses defaults for fields not present in JSON config
private func pocketMimiConfigFromJSON(_ jsonConfig: PocketMimiJSONConfig) -> PocketMimiConfig {
    let seanetConfig = PocketSeanetConfig(
        dimension: jsonConfig.seanet.dimension,
        channels: jsonConfig.channels,
        causal: true,  // Default for PocketTTS
        nfilters: jsonConfig.seanet.nFilters,
        nresidualLayers: jsonConfig.seanet.nResidualLayers,
        ratios: jsonConfig.seanet.ratios,
        ksize: jsonConfig.seanet.kernelSize,
        residualKsize: jsonConfig.seanet.residualKernelSize,
        lastKsize: jsonConfig.seanet.lastKernelSize,
        dilationBase: jsonConfig.seanet.dilationBase,
        padMode: jsonConfig.seanet.padMode == "constant" ? .constant : .constant,
        trueSkip: true,  // Default for PocketTTS
        compress: jsonConfig.seanet.compress
    )

    let transformerConfig = PocketTransformerConfig(
        dModel: jsonConfig.transformer.dModel,
        numHeads: jsonConfig.transformer.numHeads,
        numLayers: jsonConfig.transformer.numLayers,
        causal: true,  // Default
        normFirst: true,  // Default
        biasFF: false,  // Default
        biasAttn: false,  // Default
        layerScale: jsonConfig.transformer.layerScale,
        positionalEmbedding: "rope",  // Default
        useConvBlock: false,  // Default
        crossAttention: false,  // Default
        convKernelSize: 3,  // Default
        useConvBias: true,  // Default
        gating: false,  // Default
        norm: "layer_norm",  // Default
        context: jsonConfig.transformer.context,
        maxPeriod: Int(jsonConfig.transformer.maxPeriod),
        maxSeqLen: 8192,  // Default
        kvRepeat: 1,  // Default
        dimFeedforward: jsonConfig.transformer.dimFeedforward,
        convLayout: true  // Default
    )

    return PocketMimiConfig(
        channels: jsonConfig.channels,
        sampleRate: Double(jsonConfig.sampleRate),
        frameRate: Double(jsonConfig.frameRate),
        seanet: seanetConfig,
        transformer: transformerConfig
    )
}
