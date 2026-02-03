//
//  LUTConditioner.swift
//  Swift-TTS
//
//  Lookup Table Conditioner for text-to-embedding conversion.
//

import Foundation
import MLX
import MLXNN

// MARK: - LUT Conditioner

/// Lookup Table Conditioner that converts tokenized text to embeddings
public class LUTConditioner: Module, UnaryLayer {
    public let tokenizer: SentencePieceTokenizer
    public let dim: Int
    public let outputDim: Int

    private let embed: Embedding
    private let outputProj: Linear?

    public init(
        tokenizer: SentencePieceTokenizer,
        nBins: Int,
        dim: Int,
        outputDim: Int
    ) {
        self.tokenizer = tokenizer
        self.dim = dim
        self.outputDim = outputDim

        // Embedding layer: vocab_size + 1 for potential special tokens
        self.embed = Embedding(embeddingCount: nBins + 1, dimensions: dim)

        // Optional output projection if dimensions don't match
        if dim != outputDim {
            self.outputProj = Linear(dim, outputDim, bias: false)
        } else {
            self.outputProj = nil
        }

        super.init()
    }

    // MARK: - Initialization from config

    /// Create LUTConditioner from config, loading tokenizer from HuggingFace
    /// - Parameters:
    ///   - config: Lookup table configuration
    ///   - outputDim: Output embedding dimension
    ///   - repoId: HuggingFace repository ID (default: smdesai/pocket-tts)
    ///   - progressHandler: Optional download progress callback
    public static func fromConfig(
        config: LookupTableConfig,
        outputDim: Int,
        repoId: String = PocketTTSRepo.repoId,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> LUTConditioner {
        // Load tokenizer from HuggingFace repo
        let tokenizer = try await SentencePieceTokenizer(
            nBins: config.nBins,
            repoId: repoId,
            progressHandler: progressHandler
        )

        return LUTConditioner(
            tokenizer: tokenizer,
            nBins: config.nBins,
            dim: config.dim,
            outputDim: outputDim
        )
    }

    /// Create LUTConditioner from local model folder
    /// - Parameters:
    ///   - config: Lookup table configuration
    ///   - outputDim: Output embedding dimension
    ///   - modelFolder: Local folder containing tokenizer files
    public static func fromLocalFolder(
        config: LookupTableConfig,
        outputDim: Int,
        modelFolder: URL
    ) throws -> LUTConditioner {
        let tokenizer = try SentencePieceTokenizer(
            nBins: config.nBins,
            modelFolder: modelFolder
        )

        return LUTConditioner(
            tokenizer: tokenizer,
            nBins: config.nBins,
            dim: config.dim,
            outputDim: outputDim
        )
    }

    // MARK: - Text Processing

    /// Tokenize text into TokenizedText structure
    public func prepare(_ text: String) -> TokenizedText {
        return tokenizer.tokenize(text)
    }

    // MARK: - Forward Pass

    /// Convert tokenized text to embeddings
    public func callAsFunction(_ inputs: TokenizedText) -> MLXArray {
        var embeds = embed(inputs.tokens)

        if let proj = outputProj {
            embeds = proj(embeds)
        }

        return embeds
    }

    /// Forward pass with raw MLXArray input
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var embeds = embed(x)

        if let proj = outputProj {
            embeds = proj(embeds)
        }

        return embeds
    }
}

// MARK: - Text Utilities

/// Split text into sentences for better generation
public func splitIntoSentences(_ text: String, maxLength: Int = 200) -> [String] {
    let sentenceDelimiters = CharacterSet(charactersIn: ".!?")
    var sentences: [String] = []
    var currentSentence = ""

    for char in text {
        currentSentence.append(char)

        if sentenceDelimiters.contains(char.unicodeScalars.first!) {
            let trimmed = currentSentence.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                sentences.append(trimmed)
            }
            currentSentence = ""
        }
    }

    // Add any remaining text
    let trimmed = currentSentence.trimmingCharacters(in: .whitespaces)
    if !trimmed.isEmpty {
        sentences.append(trimmed)
    }

    // Combine short sentences or split long ones
    var result: [String] = []
    var buffer = ""

    for sentence in sentences {
        if buffer.isEmpty {
            buffer = sentence
        } else if (buffer.count + sentence.count + 1) <= maxLength {
            buffer += " " + sentence
        } else {
            result.append(buffer)
            buffer = sentence
        }
    }

    if !buffer.isEmpty {
        result.append(buffer)
    }

    return result
}

/// Prepare text prompt matching Python's preprocessing:
/// 1. Strip whitespace
/// 2. Replace newlines with spaces
/// 3. Capitalize first letter if needed
/// 4. Add period if text ends with alphanumeric character
/// 5. Add 8-space prefix for short text (< 5 words)
/// 6. Estimate frames after EOS based on word count
public func prepareTextPrompt(_ text: String) -> (cleanedText: String, framesAfterEos: Int) {
    var cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !cleaned.isEmpty else {
        return (cleaned, 1)
    }

    // Replace newlines and multiple spaces (matching Python)
    cleaned = cleaned.replacingOccurrences(of: "\n", with: " ")
    cleaned = cleaned.replacingOccurrences(of: "\r", with: " ")
    cleaned = cleaned.replacingOccurrences(of: "  ", with: " ")

    let wordCount = cleaned.split(separator: " ").count

    // Estimate frames after EOS (matching Python logic)
    let framesAfterEos: Int
    if wordCount <= 4 {
        framesAfterEos = 3
    } else {
        framesAfterEos = 1
    }

    // Capitalize first letter if needed (matching Python)
    if let first = cleaned.first, !first.isUppercase {
        cleaned = first.uppercased() + String(cleaned.dropFirst())
    }

    // Add period if last character is alphanumeric (matching Python: text[-1].isalnum())
    if let last = cleaned.last, last.isLetter || last.isNumber {
        cleaned = cleaned + "."
    }

    // Add 8-space prefix for short text (matching Python: " " * 8 + text)
    if wordCount < 5 {
        cleaned = String(repeating: " ", count: 8) + cleaned
    }

    return (cleaned, framesAfterEos)
}
