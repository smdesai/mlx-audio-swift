//
//  VoiceLoader.swift
//  Swift-TTS
//
//  Voice loading utilities for PocketTTS predefined and custom voices.
//

import Foundation
import HuggingFace
import MLX

// MARK: - Predefined Voices

/// Available predefined voice names
public let pocketTTSVoiceNames: [String] = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma"
]

/// Predefined voice URLs from HuggingFace Hub
public let predefinedVoices: [String: String] = {
    var voices: [String: String] = [:]
    for name in pocketTTSVoiceNames {
        voices[name] = "hf://smdesai/pocket-tts/embeddings/\(name).safetensors"
    }
    return voices
}()

// MARK: - Voice Cache

/// Cache directory for downloaded voices
private func makeVoiceCacheDirectory() -> URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("mlx_audio")
        .appendingPathComponent("pocket_tts")
        .appendingPathComponent("voices")

    try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    return cacheDir
}

// MARK: - Voice Loading

/// Load a predefined voice embedding by name
/// - Parameter voiceName: Name of the predefined voice
/// - Returns: Voice embedding tensor [1, T, D]
public func loadPredefinedVoice(_ voiceName: String) async throws -> MLXArray {
    guard let voicePath = predefinedVoices[voiceName.lowercased()] else {
        throw PocketTTSError.configurationMissing(
            "Predefined voice '\(voiceName)' not found. Available voices: \(pocketTTSVoiceNames.joined(separator: ", "))"
        )
    }

    let localPath = try await downloadIfNecessary(voicePath)
    let weights = try MLX.loadArrays(url: localPath)

    guard let audioPrompt = weights["audio_prompt"] else {
        throw PocketTTSError.weightLoadingFailed("Voice file missing 'audio_prompt' key")
    }

    return audioPrompt
}

/// Download file if not cached locally
/// - Parameter path: URL string (supports http://, https://, hf://, or local path)
/// - Returns: Local file URL
public func downloadIfNecessary(_ path: String) async throws -> URL {
    // HTTP/HTTPS URLs
    if path.hasPrefix("http://") || path.hasPrefix("https://") {
        return try await downloadFromURL(path)
    }

    // HuggingFace Hub URLs
    if path.hasPrefix("hf://") {
        return try await downloadFromHuggingFace(path)
    }

    // Local path
    return URL(fileURLWithPath: path)
}

/// Download from HTTP/HTTPS URL
private func downloadFromURL(_ urlString: String) async throws -> URL {
    let cacheDir = makeVoiceCacheDirectory()

    // Create cache filename from URL hash
    let suffix = (urlString as NSString).pathExtension
    let hash = urlString.data(using: .utf8)!.base64EncodedString()
        .replacingOccurrences(of: "/", with: "_")
        .prefix(32)
    let cachedFile = cacheDir.appendingPathComponent("\(hash).\(suffix)")

    // Return if already cached
    if FileManager.default.fileExists(atPath: cachedFile.path) {
        return cachedFile
    }

    // Download
    guard let url = URL(string: urlString) else {
        throw PocketTTSError.weightLoadingFailed("Invalid URL: \(urlString)")
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
        throw PocketTTSError.weightLoadingFailed("Download failed for: \(urlString)")
    }

    try data.write(to: cachedFile)
    return cachedFile
}

/// Download from HuggingFace Hub using HubClient
private func downloadFromHuggingFace(_ hfPath: String) async throws -> URL {
    // Parse hf:// URL
    // Format: hf://owner/repo/path/to/file@revision or hf://owner/repo/path/to/file
    var path = hfPath
    path.removeFirst(5)  // Remove "hf://"

    // Split into all components
    let components = path.split(separator: "/")
    guard components.count >= 3 else {
        // Need at least: owner, repo, filename
        throw PocketTTSError.weightLoadingFailed("Invalid HuggingFace path: \(hfPath)")
    }

    // Repo ID is first two components (owner/repo)
    let repoId = "\(components[0])/\(components[1])"

    // Filename is everything after the repo ID
    var filename = components.dropFirst(2).joined(separator: "/")

    // Extract revision if present (e.g., file.safetensors@main)
    var revision: String = "main"
    if filename.contains("@") {
        let fileParts = filename.split(separator: "@", maxSplits: 1)
        filename = String(fileParts[0])
        revision = String(fileParts[1])
    }

    // Use HuggingFace client to download
    let client = HubClient.default
    let cache = client.cache ?? HubCache.default

    guard let repoID = Repo.ID(rawValue: repoId) else {
        throw PocketTTSError.weightLoadingFailed("Invalid HuggingFace repository ID: \(repoId)")
    }

    // Use persistent cache directory based on repo ID
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDirectory = cache.cacheDirectory.appendingPathComponent(modelSubdir)
    let fileURL = modelDirectory.appendingPathComponent(filename)

    // Check if file is already cached
    if FileManager.default.fileExists(atPath: fileURL.path) {
        return fileURL
    }

    // Download the specific file
    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDirectory,
        revision: revision,
        matching: [filename]
    )

    guard FileManager.default.fileExists(atPath: fileURL.path) else {
        throw PocketTTSError.weightLoadingFailed("File not found after download: \(filename)")
    }

    return fileURL
}

// MARK: - Voice Info

/// Voice metadata
public struct VoiceInfo {
    public let name: String
    public let displayName: String
    public let description: String
    public let isPredefined: Bool

    public init(name: String, displayName: String, description: String = "", isPredefined: Bool = true) {
        self.name = name
        self.displayName = displayName
        self.description = description
        self.isPredefined = isPredefined
    }
}

/// Get info for all available predefined voices
public func getAvailableVoices() -> [VoiceInfo] {
    return [
        VoiceInfo(name: "alba", displayName: "Alba", description: "Female voice, clear and professional"),
        VoiceInfo(name: "marius", displayName: "Marius", description: "Male voice, warm and expressive"),
        VoiceInfo(name: "javert", displayName: "Javert", description: "Male voice, authoritative"),
        VoiceInfo(name: "jean", displayName: "Jean", description: "Male voice, gentle"),
        VoiceInfo(name: "fantine", displayName: "Fantine", description: "Female voice, soft"),
        VoiceInfo(name: "cosette", displayName: "Cosette", description: "Female voice, youthful"),
        VoiceInfo(name: "eponine", displayName: "Eponine", description: "Female voice, expressive"),
        VoiceInfo(name: "azelma", displayName: "Azelma", description: "Female voice, bright")
    ]
}
