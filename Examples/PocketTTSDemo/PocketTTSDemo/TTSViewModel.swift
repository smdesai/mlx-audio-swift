//
//  TTSViewModel.swift
//  PocketTTSDemo
//
//  ViewModel for PocketTTS speech synthesis.
//

import AVFoundation
import Foundation
import MLXAudioTTS
import Observation
import OSLog

/// Available voices for PocketTTS
enum PocketVoice: String, CaseIterable, Identifiable {
    case alba
    case marius
    case javert
    case jean
    case fantine
    case cosette
    case eponine
    case azelma

    var id: String { rawValue }

    var displayName: String {
        rawValue.capitalized
    }
}

/// Generation state for the TTS process
enum GenerationState: Equatable {
    case idle
    case initializing
    case generating
    case streaming(chunksPlayed: Int)
    case completed
    case error(String)

    var isGenerating: Bool {
        switch self {
        case .initializing, .generating:
            return true
        default:
            return false
        }
    }

    var isStreaming: Bool {
        switch self {
        case .streaming:
            return true
        default:
            return false
        }
    }

    var isBusy: Bool {
        switch self {
        case .initializing, .generating, .streaming:
            return true
        default:
            return false
        }
    }

    var statusText: String {
        switch self {
        case .idle:
            return "Ready"
        case .initializing:
            return "Loading models..."
        case .generating:
            return "Generating audio..."
        case .streaming(let chunks):
            return "Streaming... \(chunks) chunks"
        case .completed:
            return "Audio ready"
        case .error(let message):
            return "Error: \(message)"
        }
    }
}

/// ViewModel handling TTS synthesis, playback, and state management
@MainActor
@Observable
final class TTSViewModel {

    // MARK: - Published State

    var text: String = "Hello world, this is a test of PocketTTS speech synthesis."
    var selectedVoice: PocketVoice = .alba
    var generationState: GenerationState = .idle
    var audioURL: URL?
    var isPlaying: Bool = false

    // MARK: - Generation Stats

    var audioDuration: Double?
    var generationTime: Double?
    var realTimeFactor: Double?
    var initializationTime: Double?
    var isFirstGeneration: Bool = true

    // Streaming stats
    var streamingFirstChunkTime: Double?
    var streamingTotalTime: Double?
    var streamingChunkCount: Int?
    var streamingRTF: Double?

    // Track last operation type
    var lastOperationWasStreaming: Bool = false

    // MARK: - Private Properties

    private let logger = Logger(subsystem: "PocketTTSDemo", category: "TTSViewModel")
    private var session: PocketTTSSession?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegate?
    private var generationTask: Task<Void, Never>?

    // Streaming playback
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamingTask: Task<Void, Never>?
    private let streamingSampleRate: Double = 24000

    // MARK: - Computed Properties

    var canGenerate: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !generationState.isStreaming
    }

    var canStream: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !generationState.isGenerating
    }

    var canPlay: Bool {
        audioURL != nil && !generationState.isBusy
    }

    var canShare: Bool {
        audioURL != nil && !generationState.isBusy
    }

    var generateButtonTitle: String {
        generationState.isGenerating ? "Stop" : "Generate"
    }

    var streamButtonTitle: String {
        generationState.isStreaming ? "Stop" : "Stream"
    }

    // MARK: - Public Methods

    func generateOrStop() {
        if generationState.isGenerating {
            stopGeneration()
        } else {
            startGeneration()
        }
    }

    func streamOrStop() {
        if generationState.isStreaming {
            stopStreaming()
        } else {
            startStreaming()
        }
    }

    func play() {
        guard let url = audioURL else { return }

        // Stop any existing playback first
        stopPlayback()

        do {
            // Configure audio session for playback
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayerDelegate = AudioPlayerDelegate { [weak self] in
                Task { @MainActor in
                    self?.isPlaying = false
                }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            audioPlayer?.play()
            isPlaying = true
            logger.info("Started playback")
        } catch {
            logger.error("Playback failed: \(error.localizedDescription)")
            generationState = .error("Playback failed: \(error.localizedDescription)")
        }
    }

    func stopPlayback() {
        audioPlayer?.stop()
        isPlaying = false
    }

    func shareURL() -> URL? {
        audioURL
    }

    func clearText() {
        text = ""
    }

    // MARK: - Private Methods - Generation

    private func startGeneration() {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }

        // Cancel any existing playback
        stopPlayback()

        // Capture voice before async work
        let voice = selectedVoice.rawValue

        generationTask = Task {
            await self.performGeneration(text: trimmedText, voice: voice)
        }
    }

    private func performGeneration(text: String, voice: String) async {
        do {
            // Clear previous stats
            audioDuration = nil
            generationTime = nil
            realTimeFactor = nil

            // Initialize session if needed
            try await initializeSessionIfNeeded()

            guard let session = session else {
                generationState = .error("Session not initialized")
                return
            }

            // Set voice
            try await session.setVoice(voice)

            generationState = .generating
            logger.info("Generating audio for voice: \(voice)")

            let startTime = Date()

            // Generate audio
            let samples = try await session.generate(text: text)

            let elapsed = Date().timeIntervalSince(startTime)

            // Check for cancellation
            if Task.isCancelled {
                logger.info("Generation cancelled")
                generationState = .idle
                return
            }

            // Save to file
            let outputURL = getOutputURL()
            try saveWAV(samples: samples, sampleRate: Int(session.sampleRate), to: outputURL)
            audioURL = outputURL

            // Calculate stats
            generationTime = elapsed
            if let duration = getAudioDuration(url: outputURL) {
                audioDuration = duration
                realTimeFactor = duration / elapsed
                logger.info("Generated \(String(format: "%.2f", duration))s audio in \(String(format: "%.2f", elapsed))s (RTFx: \(String(format: "%.2f", duration / elapsed)))")
            }

            isFirstGeneration = false
            lastOperationWasStreaming = false
            generationState = .completed
            logger.info("Audio saved to: \(outputURL.lastPathComponent)")

        } catch {
            if Task.isCancelled {
                generationState = .idle
            } else {
                logger.error("Generation failed: \(error.localizedDescription)")
                generationState = .error(error.localizedDescription)
            }
        }
    }

    private func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        generationState = .idle
        logger.info("Generation stopped by user")
    }

    // MARK: - Private Methods - Streaming

    private func startStreaming() {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }

        // Cancel any existing playback
        stopPlayback()

        // Capture voice before async work
        let voice = selectedVoice.rawValue

        streamingTask = Task {
            await self.performStreaming(text: trimmedText, voice: voice)
        }
    }

    private func performStreaming(text: String, voice: String) async {
        do {
            // Clear previous streaming stats
            streamingFirstChunkTime = nil
            streamingTotalTime = nil
            streamingChunkCount = nil
            streamingRTF = nil

            // Initialize session if needed
            try await initializeSessionIfNeeded()

            guard let session = session else {
                generationState = .error("Session not initialized")
                return
            }

            // Set voice
            try await session.setVoice(voice)

            generationState = .streaming(chunksPlayed: 0)
            logger.info("Starting streaming for voice: \(voice)")

            // Set up audio engine
            try setupAudioEngine()

            let startTime = Date()
            var chunkCount = 0
            var firstChunkTime: Double?
            var allSamples: [Float] = []

            // Get the streaming synthesis
            let stream = session.generateStream(text: text)

            // Process chunks as they arrive
            for await samples in stream {
                // Check for cancellation
                if Task.isCancelled {
                    logger.info("Streaming cancelled")
                    break
                }

                // Track time to first chunk
                if chunkCount == 0 {
                    firstChunkTime = Date().timeIntervalSince(startTime)
                    logger.info("First chunk in \(String(format: "%.2f", firstChunkTime!))s")
                }

                // Schedule the audio buffer for playback
                scheduleAudioBuffer(samples: samples)
                allSamples.append(contentsOf: samples)
                chunkCount += 1
                generationState = .streaming(chunksPlayed: chunkCount)
            }

            // Wait for playback to complete
            await waitForPlaybackCompletion()

            let elapsed = Date().timeIntervalSince(startTime)
            let duration = Double(allSamples.count) / streamingSampleRate
            logger.info("Streamed \(chunkCount) chunks (\(String(format: "%.2f", duration))s) in \(String(format: "%.2f", elapsed))s")

            // Save the complete audio to file for later playback
            if !allSamples.isEmpty {
                let outputURL = getOutputURL()
                try saveWAV(samples: allSamples, sampleRate: Int(streamingSampleRate), to: outputURL)
                audioURL = outputURL
                audioDuration = duration
            }

            // Store streaming stats
            streamingFirstChunkTime = firstChunkTime
            streamingTotalTime = elapsed
            streamingChunkCount = chunkCount
            if duration > 0 {
                streamingRTF = duration / elapsed
            }

            // Clean up
            teardownAudioEngine()
            lastOperationWasStreaming = true
            generationState = .completed
            isFirstGeneration = false

        } catch {
            if Task.isCancelled {
                generationState = .idle
            } else {
                logger.error("Streaming failed: \(error.localizedDescription)")
                generationState = .error(error.localizedDescription)
            }
            teardownAudioEngine()
        }
    }

    private func stopStreaming() {
        streamingTask?.cancel()
        streamingTask = nil
        teardownAudioEngine()
        generationState = .idle
        logger.info("Streaming stopped by user")
    }

    // MARK: - Audio Engine Methods

    private func setupAudioEngine() throws {
        // Configure audio session
        try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()

        engine.attach(player)

        // PocketTTS outputs 24kHz mono float32
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: streamingSampleRate,
            channels: 1,
            interleaved: false
        )!

        engine.connect(player, to: engine.mainMixerNode, format: format)

        try engine.start()
        player.play()

        self.audioEngine = engine
        self.playerNode = player

        logger.info("Audio engine started")
    }

    private func scheduleAudioBuffer(samples: [Float]) {
        guard let player = playerNode else { return }

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: streamingSampleRate,
            channels: 1,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            logger.error("Failed to create audio buffer")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)

        // Copy samples to buffer
        if let channelData = buffer.floatChannelData?[0] {
            for (index, sample) in samples.enumerated() {
                channelData[index] = sample
            }
        }

        player.scheduleBuffer(buffer)
    }

    private func waitForPlaybackCompletion() async {
        guard let player = playerNode else { return }

        // Wait for all scheduled buffers to finish playing
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            // Schedule an empty buffer with a completion handler
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: streamingSampleRate,
                channels: 1,
                interleaved: false
            )!

            if let emptyBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1) {
                emptyBuffer.frameLength = 0
                player.scheduleBuffer(emptyBuffer) {
                    continuation.resume()
                }
            } else {
                continuation.resume()
            }
        }
    }

    private func teardownAudioEngine() {
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        logger.info("Audio engine stopped")
    }

    // MARK: - Session Initialization

    private func initializeSessionIfNeeded() async throws {
        if session == nil {
            generationState = .initializing
            logger.info("Initializing PocketTTS session...")

            let initStart = Date()

            // Download config and initialize
            let modelDirectory = try await downloadPocketTTSModel(
                repoId: PocketTTSRepo.repoId,
                matching: ["*.safetensors", "*.json", "tokenizer.model"]
            )

            let configURL = modelDirectory.appendingPathComponent("config.json")
            let config = try PocketTTSModelConfig.load(from: configURL)

            let newSession = PocketTTSSession()
            try await newSession.loadModel(config: config, repoId: PocketTTSRepo.repoId)

            session = newSession
            initializationTime = Date().timeIntervalSince(initStart)
            logger.info("Initialization completed in \(String(format: "%.2f", self.initializationTime!))s")
        }
    }

    // MARK: - Utility Methods

    private func getOutputURL() -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let timestamp = Int(Date().timeIntervalSince1970)
        return documentsPath.appendingPathComponent("pockettts_\(timestamp).wav")
    }

    private func getAudioDuration(url: URL) -> Double? {
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            return player.duration
        } catch {
            logger.error("Failed to get audio duration: \(error.localizedDescription)")
            return nil
        }
    }

    private func saveWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        // WAV header
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count * 2) // 16-bit samples

        var data = Data()

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: (36 + dataSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) }) // PCM
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Sample = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: int16Sample.littleEndian) { Array($0) })
        }

        try data.write(to: url)
    }
}

// MARK: - Audio Player Delegate

private final class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate, Sendable {
    private let onFinished: @Sendable () -> Void

    init(onFinished: @escaping @Sendable () -> Void) {
        self.onFinished = onFinished
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinished()
    }
}
