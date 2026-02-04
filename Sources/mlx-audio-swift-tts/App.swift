import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepositoryID(String)
    case unsupportedModelType(String?)
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? {
        description
    }

    var description: String {
        switch self {
        case .invalidRepositoryID(let model):
            "Invalid repository ID: \(model)"
        case .unsupportedModelType(let modelType):
            "Unsupported model type: \(String(describing: modelType))"
        case .failedToCreateAudioBuffer:
            "Failed to create audio buffer"
        case .failedToAccessAudioBufferData:
            "Failed to access audio buffer data"
        }
    }
}

@main
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                model: args.model,
                text: args.text,
                voice: args.voice,
                voiceFile: args.voiceFile,
                voiceEmbedding: args.voiceEmbedding,
                exportVoice: args.exportVoice,
                outputPath: args.outputPath,
                refAudioPath: args.refAudioPath,
                refText: args.refText,
                maxTokens: args.maxTokens,
                temperature: args.temperature,
                topP: args.topP
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        model: String,
        text: String?,
        voice: String?,
        voiceFile: String?,
        voiceEmbedding: String?,
        exportVoice: String?,
        outputPath: String?,
        refAudioPath: String?,
        refText: String?,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        print("Loading model (\(model))")

        // Check for HF token in environment (macOS) or Info.plist (iOS) as a fallback
        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"] ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let loadedModel: SpeechGenerationModel
        do {
            loadedModel = try await TTSModelUtils.loadModel(modelRepo: model, hfToken: hfToken)
        } catch let error as TTSModelUtilsError {
            switch error {
            case .invalidRepositoryID(let modelRepo):
                throw AppError.invalidRepositoryID(modelRepo)
            case .unsupportedModelType(let modelType):
                throw AppError.unsupportedModelType(modelType)
            }
        }

        // Handle voice embedding export for PocketTTS models
        if let exportVoicePath = exportVoice, let pocketModel = loadedModel as? PocketTTSModel {
            let audioURL = resovleURL(path: exportVoicePath)
            let outputURL = makeOutputURL(outputPath: outputPath, defaultExtension: "safetensors")
            print("Exporting voice embedding from: \(audioURL.path)")
            let started = CFAbsoluteTimeGetCurrent()
            try pocketModel.exportVoiceEmbedding(from: audioURL, to: outputURL)
            let elapsed = CFAbsoluteTimeGetCurrent() - started
            print("Exported voice embedding to: \(outputURL.path)")
            print(String(format: "Export completed in %.2fs", elapsed))
            print("Memory usage:\n\(Memory.snapshot())")
            return
        }

        // Text is required for generation
        guard let text = text, !text.isEmpty else {
            throw CLIError.missingValue("--text")
        }

        print("Generating")
        let started = CFAbsoluteTimeGetCurrent()

        let refAudio: MLXArray?
        if let refAudioPath, !refAudioPath.isEmpty {
            let refAudioURL = resovleURL(path: refAudioPath)
            (_, refAudio) = try loadAudioArray(from: refAudioURL)
        } else {
            refAudio = nil
        }

        // Handle voice cloning/embedding for PocketTTS models
        let audioData: [Float]
        if let pocketModel = loadedModel as? PocketTTSModel {
            let state: PocketTTSState
            if let voiceEmbeddingPath = voiceEmbedding {
                // Use pre-exported voice embedding (memory efficient ~700MB)
                let embeddingURL = resovleURL(path: voiceEmbeddingPath)
                print("Loading voice embedding from: \(embeddingURL.path)")
                state = try pocketModel.getStateForVoiceEmbedding(embeddingURL)
            } else if let voiceFilePath = voiceFile {
                // Clone voice from audio file (higher memory ~1.8GB)
                let voiceURL = resovleURL(path: voiceFilePath)
                print("Loading voice from: \(voiceURL.path)")
                state = try pocketModel.getStateForAudioFile(voiceURL)
            } else if let voiceName = voice {
                // Use predefined voice
                state = try await pocketModel.getStateForVoice(voiceName)
            } else {
                // Default voice
                state = try await pocketModel.getStateForVoice("alba")
            }
            pocketModel.temp = temperature
            let audio = pocketModel.generateAudio(state: state, text: text)
            audioData = audio.asArray(Float.self)
        } else {
            audioData = try await loadedModel.generate(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: nil,
                generationParameters: GenerateParameters(
                    maxTokens: maxTokens,
                    temperature: temperature,
                    topP: topP
                )
            ).asArray(Float.self)
        }

        let outputURL = makeOutputURL(outputPath: outputPath, defaultExtension: "wav")
        let sampleRate = Double(loadedModel.sampleRate)
        try writeWavFile(samples: audioData, sampleRate: sampleRate, outputURL: outputURL)
        print("Wrote WAV to \(outputURL.path)")

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        let audioDuration = Double(audioData.count) / sampleRate
        let rtfx = audioDuration / elapsed

        print(String(format: "Finished generation in %.2fs", elapsed))
        print(String(format: "Audio duration: %.2fs | RTFx: %.1fx", audioDuration, rtfx))
        print("Memory usage:\n\(Memory.snapshot())")
    }

    private static func makeOutputURL(outputPath: String?, defaultExtension: String = "wav") -> URL {
        let defaultName = "output.\(defaultExtension)"
        let outputName = outputPath?.isEmpty == false ? outputPath! : defaultName
        if outputName.hasPrefix("/") {
            return URL(fileURLWithPath: outputName)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(outputName)
    }

    private static func resovleURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    private static func writeWavFile(samples: [Float], sampleRate: Double, outputURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AppError.failedToCreateAudioBuffer
        }
        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData else {
            throw AppError.failedToAccessAudioBufferData
        }
        for i in 0 ..< samples.count {
            channelData[0][i] = samples[i]
        }
        let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try audioFile.write(from: buffer)
    }
}

// MARK: -

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let k): "Missing value for \(k)"
        case .unknownOption(let k): "Unknown option \(k)"
        case .invalidValue(let k, let v): "Invalid value for \(k): \(v)"
        }
    }
}

struct CLI {
    let model: String
    let text: String?
    let voice: String?
    let voiceFile: String?
    let voiceEmbedding: String?  // Pre-exported voice embedding (.safetensors)
    let exportVoice: String?     // Audio file to export as embedding
    let outputPath: String?
    let refAudioPath: String?
    let refText: String?
    let maxTokens: Int
    let temperature: Float
    let topP: Float

    static func parse() throws -> CLI {
        var text: String?
        var voice: String? = nil
        var voiceFile: String? = nil
        var voiceEmbedding: String? = nil
        var exportVoice: String? = nil
        var outputPath: String? = nil
        var model = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"
        var refAudioPath: String? = nil
        var refText: String? = nil
        var maxTokens: Int = 1200
        var temperature: Float = 0.7
        var topP: Float = 0.9

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--text", "-t":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                text = v
            case "--voice", "-v":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voice = v
            case "--voice-file", "-vf":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voiceFile = v
            case "--voice-embedding", "-ve":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voiceEmbedding = v
            case "--export-voice":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                exportVoice = v
            case "--model":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                model = v
            case "--output", "-o":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                outputPath = v
            case "--ref_audio":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refAudioPath = v
            case "--ref_text":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refText = v
            case "--max_tokens":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                maxTokens = value
            case "--temperature":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                temperature = value
            case "--top_p":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                topP = value
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if text == nil, !arg.hasPrefix("-") {
                    text = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        // Text is required unless exporting a voice
        if exportVoice == nil {
            guard let finalText = text, !finalText.isEmpty else {
                throw CLIError.missingValue("--text")
            }
        }

        return CLI(
            model: model,
            text: text,
            voice: voice,
            voiceFile: voiceFile,
            voiceEmbedding: voiceEmbedding,
            exportVoice: exportVoice,
            outputPath: outputPath,
            refAudioPath: refAudioPath,
            refText: refText,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-tts"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice conversational_b] [--model <hf-repo>] [--output <path>] [--ref_audio <path>] [--ref_text <string>] [--max_tokens <int>] [--temperature <float>] [--top_p <float>]

        Options:
          -t, --text <string>           Text to synthesize (required if not passed as trailing arg)
          -v, --voice <name>            Voice id (predefined voice name)
          -vf, --voice-file <path>      Path to audio file for voice cloning (PocketTTS only, WAV/MP3/FLAC)
          -ve, --voice-embedding <path> Path to pre-exported voice embedding (.safetensors)
              --model <repo>            HF repo id. Default: Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit
          -o, --output <path>           Output WAV path. Default: ./output.wav
              --ref_audio <path>        Path to reference audio
              --ref_text <string>       Caption for reference audio
              --max_tokens <int>        Maximum number of tokens to generate. Default: 1200
              --temperature <float>     Sampling temperature. Default: 0.7
              --top_p <float>           Top-p sampling. Default: 0.9
          -h, --help                    Show this help

        Voice Cloning (PocketTTS):
          Use --voice-file to clone a voice from an audio file. The audio will be
          truncated to 10 seconds (to save memory) and resampled to 24kHz if needed.
          Example: \(exe) --model smdesai/pocket-tts --text "Hello" --voice-file voice.wav

        Pre-exported Voice Embeddings (memory efficient):
          For lower memory usage (~700MB vs ~1.8GB), pre-export voice embeddings:
          1. Export: \(exe) --model smdesai/pocket-tts --export-voice input.wav --output voice.safetensors
          2. Use:    \(exe) --model smdesai/pocket-tts --text "Hello" --voice-embedding voice.safetensors
        """)
    }
}
