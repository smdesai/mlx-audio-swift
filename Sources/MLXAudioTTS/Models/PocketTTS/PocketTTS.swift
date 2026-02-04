//
//  PocketTTS.swift
//  Swift-TTS
//
//  Main PocketTTS model implementation for text-to-speech generation.
//

import Foundation
import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

// MARK: - Model State

/// State container for PocketTTS generation (class for reference semantics in closures)
public class PocketTTSState {
    public var flowCache: [KVCacheSimple]

    public init(flowCache: [KVCacheSimple]) {
        self.flowCache = flowCache
    }
}

// MARK: - PocketTTS Model

/// PocketTTS: Flow-matching TTS model combining transformer backbone with Mimi codec
public class PocketTTSModel: Module, @unchecked Sendable {
    public let config: PocketTTSModelConfig
    public let flowLM: FlowLMModel
    public let mimi: PocketMimiAdapter

    // Generation parameters
    public var temp: Float = PocketTTSDefaults.temperature
    public var lsdDecodeSteps: Int = PocketTTSDefaults.lsdDecodeSteps
    public var noiseClamp: Float? = PocketTTSDefaults.noiseClamp
    public var eosThreshold: Float = PocketTTSDefaults.eosThreshold
    public var minStepsBeforeEOS: Int = PocketTTSDefaults.minStepsBeforeEOS

    // Speaker projection for voice conditioning
    public var speakerProjWeight: MLXArray

    public init(config: PocketTTSModelConfig, flowLM: FlowLMModel, mimi: PocketMimiAdapter) {
        guard let flowLMConfig = config.flowLM, let mimiConfig = config.mimi else {
            fatalError("PocketTTS requires flow_lm and mimi config sections.")
        }

        self.config = config
        self.flowLM = flowLM
        self.mimi = mimi

        // Initialize speaker projection weight
        self.speakerProjWeight = MLXArray.zeros([
            flowLMConfig.transformer.dModel,
            mimiConfig.quantizer.outputDimension
        ])

        super.init()
    }

    // MARK: - Properties

    /// Optional sample rate accessor (for internal use)
    public var optionalSampleRate: Int? {
        return config.mimi?.sampleRate
    }

    public var modelType: String {
        return config.modelType
    }

    // MARK: - State Management

    /// Initialize fresh generation state
    public func initState() -> PocketTTSState {
        return PocketTTSState(flowCache: flowLM.makeCache())
    }

    // MARK: - Audio Encoding

    /// Encode audio prompt to conditioning
    private func encodeAudio(_ audio: MLXArray) -> MLXArray {
        let encoded = mimi.encodeToLatent(audio)

        // Transpose: [B, D, T] -> [B, T, D]
        let latents = encoded.transposed(0, 2, 1).asType(.float32)

        // Project to transformer dimension
        let conditioning = MLX.matmul(latents, speakerProjWeight.transposed())
        return conditioning
    }

    // MARK: - Flow LM Execution

    /// Run flow LM to generate next latent
    private func runFlowLM(
        state: PocketTTSState,
        textTokens: MLXArray,
        backboneInputLatents: MLXArray,
        audioConditioning: MLXArray
    ) -> (outputEmbeddings: MLXArray, isEOS: MLXArray) {
        // Get text embeddings from conditioner
        let textEmbeddings = flowLM.conditioner(TokenizedText(tokens: textTokens))

        // Concatenate text with audio conditioning
        let combinedEmbeddings = MLX.concatenated([textEmbeddings, audioConditioning], axis: 1)

        // Sample next latent
        let (outputEmbeddings, isEOS) = flowLM.sampleNextLatent(
            sequence: backboneInputLatents,
            textEmbeddings: combinedEmbeddings,
            cache: state.flowCache,
            lsdDecodeSteps: lsdDecodeSteps,
            temp: temp,
            noiseClamp: noiseClamp,
            eosThreshold: eosThreshold
        )

        // Expand for sequence dimension: [B, D] -> [B, 1, D]
        return (outputEmbeddings.reshaped([outputEmbeddings.shape[0], 1, outputEmbeddings.shape[1]]), isEOS)
    }

    /// Run flow LM with default empty inputs
    private func runFlowLMAndIncrementStep(
        state: PocketTTSState,
        textTokens: MLXArray? = nil,
        backboneInputLatents: MLXArray? = nil,
        audioConditioning: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let tokens = textTokens ?? MLXArray.zeros([1, 0]).asType(.int32)
        // Use EMPTY array [1, 0, ldim] to match Python behavior
        // Python: mx.full((1, 0, self.flow_lm.ldim), float("NaN"), dtype=mx.float32)
        // Empty backbone during text processing ensures KV cache has correct positions
        let latents = backboneInputLatents ?? MLXArray.zeros([1, 0, flowLM.ldim]).asType(.float32)
        let conditioning = audioConditioning ?? MLXArray.zeros([1, 0, flowLM.dim]).asType(.float32)

        return runFlowLM(
            state: state,
            textTokens: tokens,
            backboneInputLatents: latents,
            audioConditioning: conditioning
        )
    }

    // MARK: - Cache Management

    /// Slice flow cache to keep only first N frames
    /// Note: KVCacheSimple doesn't support direct slicing, so this is a no-op for now
    private func sliceFlowCache(state: PocketTTSState, numFrames: Int) {
        // KVCacheSimple manages its own state internally
        // Slicing would require a custom cache implementation
    }

    /// Expand flow cache capacity
    /// Note: KVCacheSimple auto-expands, so this is a no-op
    private func expandFlowCache(state: PocketTTSState, sequenceLength: Int) {
        // KVCacheSimple automatically expands as needed
    }

    // MARK: - Voice Conditioning

    /// Get state for voice cloning from an audio file
    /// - Parameters:
    ///   - audioURL: URL to audio file (WAV, MP3, FLAC supported via AVFoundation)
    ///   - truncate: Whether to truncate to 30 seconds max (default: true)
    /// - Returns: PocketTTSState primed with the speaker's voice characteristics
    public func getStateForAudioFile(_ audioURL: URL, truncate: Bool = true) throws -> PocketTTSState {
        // 1. Load audio file
        let (originalSampleRate, audioData) = try loadAudioArray(from: audioURL)
        var samples = audioData.asArray(Float.self)

        // 2. Truncate to 30 seconds if requested
        if truncate {
            samples = truncateAudioSamples(samples, maxSeconds: 30.0, sampleRate: originalSampleRate)
        }

        // 3. Resample to model sample rate (24kHz)
        let targetSampleRate = Double(optionalSampleRate ?? 24000)
        if Double(originalSampleRate) != targetSampleRate {
            samples = resampleAudioLinear(samples, from: Double(originalSampleRate), to: targetSampleRate)
        }

        // 4. Shape to [1, 1, T] for encoder
        let audio = MLXArray(samples).reshaped([1, 1, samples.count])

        // 5. Encode to conditioning
        let conditioning = encodeAudio(audio)

        // 6. Return primed state
        return getStateForAudioPrompt(audioConditioning: conditioning)
    }

    /// Get state initialized with audio prompt conditioning
    public func getStateForAudioPrompt(audioConditioning: MLXArray) -> PocketTTSState {
        var state = initState()

        // Run flow LM once with audio conditioning to prime the cache
        _ = runFlowLMAndIncrementStep(
            state: state,
            audioConditioning: audioConditioning
        )

        // Optionally slice cache to conditioning length
        // sliceFlowCache(state: state, numFrames: audioConditioning.shape[1])

        return state
    }

    /// Get state for predefined voice name
    public func getStateForVoice(_ voiceName: String) async throws -> PocketTTSState {
        let voiceEmbedding = try await loadPredefinedVoice(voiceName)
        return getStateForAudioPrompt(audioConditioning: voiceEmbedding)
    }

    // MARK: - Audio Generation

    /// Generate audio from text (blocking, returns full audio)
    public func generateAudio(
        state: PocketTTSState,
        text: String,
        framesAfterEOS: Int? = nil
    ) -> MLXArray {
        var chunks: [MLXArray] = []

        for chunk in generateAudioStream(state: state, text: text, framesAfterEOS: framesAfterEOS) {
            chunks.append(chunk)
        }

        if chunks.isEmpty {
            return MLXArray.zeros([0]).asType(.float32)
        }

        return MLX.concatenated(chunks, axis: 0)
    }

    /// Generate audio stream from text
    public func generateAudioStream(
        state: PocketTTSState,
        text: String,
        framesAfterEOS: Int? = nil
    ) -> AnySequence<MLXArray> {
        let sentences = splitIntoSentences(text)

        return AnySequence { () -> AnyIterator<MLXArray> in
            var sentenceIndex = 0
            var currentGenerator: AnyIterator<MLXArray>?

            return AnyIterator {
                while true {
                    // Try to get next chunk from current generator
                    if let gen = currentGenerator, let chunk = gen.next() {
                        return chunk
                    }

                    // Move to next sentence
                    if sentenceIndex >= sentences.count {
                        return nil
                    }

                    let sentence = sentences[sentenceIndex]
                    sentenceIndex += 1

                    // Preprocess text: capitalize first letter, add period if needed (matching Python)
                    let (cleanedText, estimatedFrames) = prepareTextPrompt(sentence)
                    let actualFrames = framesAfterEOS ?? (estimatedFrames + 2)

                    currentGenerator = AnyIterator(self.generateAudioStreamShortText(
                        state: state,
                        text: cleanedText,  // Use preprocessed text with period added
                        framesAfterEOS: actualFrames
                    ).makeIterator())
                }
            }
        }
    }

    /// Generate audio stream for short text segment
    private func generateAudioStreamShortText(
        state: PocketTTSState,
        text: String,
        framesAfterEOS: Int
    ) -> AnySequence<MLXArray> {
        // Reset Mimi decoder state
        mimi.resetState()

        // Expand cache for generation
        expandFlowCache(state: state, sequenceLength: 1000)

        // Estimate generation length
        let wordCount = text.split(separator: " ").count
        let genLenSec = Float(wordCount) * 1.0 + 2.0
        let maxGenLen = Int(genLenSec * mimi.frameRate)

        // Tokenize text (text should already be preprocessed with period by prepareTextPrompt)
        let prepared = flowLM.conditioner.prepare(text)

        return AnySequence { () -> AnyIterator<MLXArray> in
            var step = 0
            var eosStep: Int? = nil
            var backboneInput = MLXArray.full([1, 1, self.flowLM.ldim], values: MLXArray(Float.nan)).asType(.float32)

            // Initial step with text tokens
            var hasRunInitial = false

            return AnyIterator {
                // Run initial step with text tokens
                if !hasRunInitial {
                    _ = self.runFlowLMAndIncrementStep(
                        state: state,
                        textTokens: prepared.tokens
                    )
                    hasRunInitial = true
                }

                guard step < maxGenLen else { return nil }

                // Check for EOS termination
                if let eos = eosStep, step >= eos + framesAfterEOS {
                    return nil
                }

                // Generate next latent
                let (nextLatent, isEOS) = self.runFlowLMAndIncrementStep(
                    state: state,
                    backboneInputLatents: backboneInput
                )

                // Check EOS (only after minimum steps to prevent early cutoff)
                if isEOS.item(Bool.self) && eosStep == nil && step >= self.minStepsBeforeEOS {
                    eosStep = step
                }

                // Denormalize latent
                let denormalized = nextLatent * self.flowLM.embStd + self.flowLM.embMean

                // Quantize through Mimi quantizer: [B, 1, D] -> [B, D, 1]
                let quantizerInput = denormalized.transposed(0, 2, 1)
                let quantized = self.mimi.quantizer(quantizerInput)

                // Decode to audio
                let audioChunk = self.mimi.decodeStep(quantized)
                let audioSlice = audioChunk[0, 0]

                // Update backbone input for next step
                backboneInput = nextLatent

                step += 1

                // Return audio: [B, C, T] -> [T]
                return audioSlice
            }
        }
    }

    // MARK: - Factory

    /// Create PocketTTS from configuration, loading tokenizer from HuggingFace
    /// - Parameters:
    ///   - config: Model configuration
    ///   - mimi: PocketMimi codec for audio decoding
    ///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
    ///   - progressHandler: Optional download progress callback
    public static func fromConfig(
        _ config: PocketTTSModelConfig,
        mimi: PocketMimi,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> PocketTTSModel {
        guard let flowLMConfig = config.flowLM, let mimiConfig = config.mimi else {
            throw PocketTTSError.configurationMissing("flow_lm and mimi configurations required")
        }

        // Create FlowLM with tokenizer from HuggingFace
        let latentDim = mimiConfig.quantizer.dimension
        let flowLM = try await FlowLMModel.fromConfig(
            flowLMConfig,
            latentDim: latentDim,
            repoId: repoId,
            progressHandler: progressHandler
        )

        // Create Mimi adapter from existing codec
        // inputDimension = FlowLM latent dim, outputDimension = Mimi decoder input dim
        let mimiAdapter = PocketMimiAdapter.fromMimi(mimi, inputDimension: latentDim, outputDimension: mimiConfig.quantizer.outputDimension)

        return PocketTTSModel(config: config, flowLM: flowLM, mimi: mimiAdapter)
    }

    /// Create PocketTTS from local model folder
    public static func fromLocalFolder(
        _ config: PocketTTSModelConfig,
        mimi: PocketMimi,
        modelFolder: URL
    ) throws -> PocketTTSModel {
        guard let flowLMConfig = config.flowLM, let mimiConfig = config.mimi else {
            throw PocketTTSError.configurationMissing("flow_lm and mimi configurations required")
        }

        let latentDim = mimiConfig.quantizer.dimension
        let flowLM = try FlowLMModel.fromLocalFolder(
            flowLMConfig,
            latentDim: latentDim,
            modelFolder: modelFolder
        )

        let mimiAdapter = PocketMimiAdapter.fromMimi(mimi, inputDimension: latentDim, outputDimension: mimiConfig.quantizer.outputDimension)

        return PocketTTSModel(config: config, flowLM: flowLM, mimi: mimiAdapter)
    }

    /// Load PocketTTS model from HuggingFace repository
    public static func fromPretrained(_ repoId: String = PocketTTSRepo.repoId) async throws -> PocketTTSModel {
        // Check if repoId is a local path
        if repoId.hasPrefix("/") || repoId.hasPrefix("./") {
            return try fromLocalPath(repoId)
        }

        let model = try await loadPocketTTSFromHub(
            repoId: repoId,
            mimi: createDefaultMimi(),
            progressHandler: { _ in }
        )
        return model
    }

    /// Load PocketTTS model from local path
    public static func fromLocalPath(_ path: String) throws -> PocketTTSModel {
        let modelFolder = URL(fileURLWithPath: path)
        let configURL = modelFolder.appendingPathComponent("config.json")
        let weightsURL = modelFolder.appendingPathComponent("model.safetensors")

        // Load config
        let config = try PocketTTSModelConfig.load(from: configURL)

        // Create Mimi codec
        let mimi = createDefaultMimi()

        // Create model
        let model = try PocketTTSModel.fromLocalFolder(config, mimi: mimi, modelFolder: modelFolder)

        // Load weights
        try loadPocketTTSWeights(model: model, from: weightsURL, strict: false)

        return model
    }

    /// Create default PocketMimi codec with standard config
    private static func createDefaultMimi() -> PocketMimi {
        // Standard PocketTTS Mimi config
        let seanetConfig = PocketSeanetConfig(
            dimension: 512,
            channels: 1,
            causal: true,
            nfilters: 64,
            nresidualLayers: 1,
            ratios: [6, 5, 4],
            ksize: 7,
            residualKsize: 3,
            lastKsize: 3,
            dilationBase: 2,
            padMode: .constant,
            trueSkip: true,
            compress: 2
        )

        let transformerConfig = PocketTransformerConfig(
            dModel: 512,
            numHeads: 8,
            numLayers: 2,
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
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedforward: 2048,
            convLayout: true
        )

        let mimiConfig = PocketMimiConfig(
            channels: 1,
            sampleRate: 24000,
            frameRate: 12.5,
            seanet: seanetConfig,
            transformer: transformerConfig
        )

        return PocketMimi(cfg: mimiConfig)
    }
}

// MARK: - SpeechGenerationModel Conformance

extension PocketTTSModel: SpeechGenerationModel {
    /// Sample rate for audio output (required by SpeechGenerationModel protocol)
    public var sampleRate: Int {
        return config.mimi?.sampleRate ?? 24000
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // Initialize state with voice if provided
        var state: PocketTTSState
        if let voiceName = voice {
            state = try await getStateForVoice(voiceName)
        } else {
            state = try await getStateForVoice(PocketTTSDefaults.defaultVoice)
        }

        // Set generation parameters
        self.temp = generationParameters.temperature

        // Generate audio
        let audio = generateAudio(state: state, text: text)

        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Initialize state with voice if provided
                    var state: PocketTTSState
                    if let voiceName = voice {
                        state = try await self.getStateForVoice(voiceName)
                    } else {
                        state = try await self.getStateForVoice(PocketTTSDefaults.defaultVoice)
                    }

                    // Set generation parameters
                    self.temp = generationParameters.temperature

                    // Stream audio chunks
                    for chunk in self.generateAudioStream(state: state, text: text) {
                        continuation.yield(.audio(chunk))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
