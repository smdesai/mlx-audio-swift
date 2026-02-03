//
//  SentencePieceTokenizer.swift
//  Swift-TTS
//
//  Native SentencePiece tokenizer for PocketTTS text processing.
//  Uses tokenizer.model from HuggingFace via SentencePiece.xcframework.
//

import Foundation
import Hub
import MLX

// MARK: - HuggingFace Repository

public enum PocketTTSRepo {
    public static let repoId = "smdesai/pocket-tts"
    public static let tokenizerModelFile = "tokenizer.model"
    public static let modelFile = "model.safetensors"
    public static let configFile = "config.json"
}

// MARK: - Tokenized Text Result

public struct TokenizedText {
    public let tokens: MLXArray

    public init(tokens: MLXArray) {
        self.tokens = tokens
    }

    public init(tokenIds: [Int]) {
        self.tokens = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, tokenIds.count])
    }
}

// MARK: - SentencePiece Tokenizer Protocol

public protocol PocketTokenizer {
    var vocabSize: Int { get }
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
}

// MARK: - C Bridge Function Declarations

// Create and load a SentencePiece model
@_silgen_name("sentencepiece_create")
fileprivate func sentencepiece_create(_ modelPath: UnsafePointer<CChar>) -> OpaquePointer?

// Encode text to IDs
@_silgen_name("sentencepiece_encode_as_ids")
fileprivate func sentencepiece_encode_as_ids(_ processor: OpaquePointer,
                                            _ text: UnsafePointer<CChar>,
                                            _ ids: UnsafeMutablePointer<UnsafeMutablePointer<Int32>?>) -> Int32

// Get vocabulary size
@_silgen_name("sentencepiece_get_piece_size")
fileprivate func sentencepiece_get_piece_size(_ processor: OpaquePointer) -> Int32

// Decode IDs back to text
@_silgen_name("sentencepiece_decode_ids")
fileprivate func sentencepiece_decode_ids(_ processor: OpaquePointer, _ ids: UnsafePointer<Int32>, _ numIds: Int32) -> UnsafeMutablePointer<CChar>?

// Clean up
@_silgen_name("sentencepiece_destroy")
fileprivate func sentencepiece_destroy(_ processor: OpaquePointer)

@_silgen_name("sentencepiece_free_ids")
fileprivate func sentencepiece_free_ids(_ ids: UnsafeMutablePointer<Int32>)

// MARK: - Native SentencePiece Tokenizer Implementation

public class SentencePieceTokenizer: PocketTokenizer {
    private let processor: OpaquePointer
    public let vocabSize: Int
    private let nBins: Int

    /// Initialize tokenizer from HuggingFace repository
    /// - Parameters:
    ///   - nBins: Expected vocabulary size
    ///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
    ///   - progressHandler: Optional progress callback for download
    public init(
        nBins: Int,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws {
        self.nBins = nBins

        // Download tokenizer.model from HuggingFace Hub
        let hub = HubApi.shared
        let repo = Hub.Repo(id: repoId)

        // Download the tokenizer.model file
        let modelFolder: URL
        if let handler = progressHandler {
            modelFolder = try await hub.snapshot(
                from: repo,
                matching: [PocketTTSRepo.tokenizerModelFile],
                progressHandler: handler
            )
        } else {
            modelFolder = try await hub.snapshot(
                from: repo,
                matching: [PocketTTSRepo.tokenizerModelFile]
            )
        }

        let tokenizerModelPath = modelFolder.appendingPathComponent(PocketTTSRepo.tokenizerModelFile)

        // Verify file exists
        guard FileManager.default.fileExists(atPath: tokenizerModelPath.path) else {
            throw PocketTTSError.tokenizerError("tokenizer.model not found at: \(tokenizerModelPath.path)")
        }

        // Load using native SentencePiece
        guard let proc = sentencepiece_create(tokenizerModelPath.path) else {
            throw PocketTTSError.tokenizerError("Failed to load SentencePiece model from: \(tokenizerModelPath.path)")
        }

        self.processor = proc

        // Get actual vocab size from loaded model
        let actualVocabSize = Int(sentencepiece_get_piece_size(processor))
        self.vocabSize = actualVocabSize
    }

    /// Initialize tokenizer from local model file path
    /// - Parameters:
    ///   - nBins: Expected vocabulary size
    ///   - modelPath: Path to tokenizer.model file
    public init(nBins: Int, modelPath: String) throws {
        self.nBins = nBins

        // Verify tokenizer.model exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw PocketTTSError.tokenizerError("tokenizer.model not found at: \(modelPath)")
        }

        // Load using native SentencePiece
        guard let proc = sentencepiece_create(modelPath) else {
            throw PocketTTSError.tokenizerError("Failed to load SentencePiece model from: \(modelPath)")
        }

        self.processor = proc

        // Get actual vocab size from loaded model
        let actualVocabSize = Int(sentencepiece_get_piece_size(processor))
        self.vocabSize = actualVocabSize
    }

    /// Initialize tokenizer from local model folder
    /// - Parameters:
    ///   - nBins: Expected vocabulary size
    ///   - modelFolder: Local folder containing tokenizer.model
    public init(nBins: Int, modelFolder: URL) throws {
        self.nBins = nBins

        // Verify tokenizer.model exists
        let tokenizerPath = modelFolder.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw PocketTTSError.tokenizerError("tokenizer.model not found at: \(modelFolder.path)")
        }

        // Load using native SentencePiece
        guard let proc = sentencepiece_create(tokenizerPath.path) else {
            throw PocketTTSError.tokenizerError("Failed to load SentencePiece model from: \(tokenizerPath.path)")
        }

        self.processor = proc

        // Get actual vocab size from loaded model
        let actualVocabSize = Int(sentencepiece_get_piece_size(processor))
        self.vocabSize = actualVocabSize
    }

    deinit {
        sentencepiece_destroy(processor)
    }

    public func encode(_ text: String) -> [Int] {
        var idsPtr: UnsafeMutablePointer<Int32>?
        let count = sentencepiece_encode_as_ids(processor, text, &idsPtr)

        guard count > 0, let ids = idsPtr else {
            return []
        }

        let result = Array(UnsafeBufferPointer(start: ids, count: Int(count)))
            .map { Int($0) }

        sentencepiece_free_ids(ids)
        return result
    }

    public func decode(_ tokens: [Int]) -> String {
        let ids32 = tokens.map { Int32($0) }
        guard let decoded = sentencepiece_decode_ids(processor, ids32, Int32(ids32.count)) else {
            return ""
        }
        defer { free(decoded) }
        return String(cString: decoded)
    }

    public func tokenize(_ text: String) -> TokenizedText {
        let tokenIds = encode(text)
        return TokenizedText(tokenIds: tokenIds)
    }
}

// MARK: - Simple Tokenizer Fallback

/// A simple fallback tokenizer for testing when SentencePiece is not available
public class SimpleTokenizer: PocketTokenizer {
    private var vocab: [String: Int] = [:]
    private var reverseVocab: [Int: String] = [:]
    public let vocabSize: Int

    public init(vocabSize: Int = 10000) {
        self.vocabSize = vocabSize

        // Build a simple character-level vocabulary
        var idx = 0
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-" {
            vocab[String(char)] = idx
            reverseVocab[idx] = String(char)
            idx += 1
        }
        vocab["<unk>"] = idx
        reverseVocab[idx] = "<unk>"
    }

    public func encode(_ text: String) -> [Int] {
        var tokens: [Int] = []
        for char in text {
            if let tokenId = vocab[String(char)] {
                tokens.append(tokenId)
            } else {
                tokens.append(vocab["<unk>"]!)
            }
        }
        return tokens
    }

    public func decode(_ tokens: [Int]) -> String {
        return tokens.compactMap { reverseVocab[$0] }.joined()
    }
}
