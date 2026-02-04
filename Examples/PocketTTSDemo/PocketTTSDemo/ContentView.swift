//
//  ContentView.swift
//  PocketTTSDemo
//
//  Main UI for PocketTTS demo app.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var viewModel = TTSViewModel()
    @State private var showingShareSheet = false
    @State private var showingFilePicker = false
    @State private var showingSavedVoicePicker = false
    @State private var showingExportAlert = false
    @State private var exportVoiceName = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Main content area
                ScrollView {
                    VStack(spacing: 24) {
                        // Voice Selection
                        voiceSelectionSection

                        // Text Input
                        textInputSection

                        // Status indicator
                        statusSection
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 100) // Space for bottom buttons
                }
                .scrollDismissesKeyboard(.interactively)

                // Bottom action buttons
                actionButtonsSection
            }
            .background(Color(.systemGroupedBackground))
            .contentShape(Rectangle())
            .onTapGesture {
                hideKeyboard()
            }
            .navigationTitle("PocketTTS")
            .navigationBarTitleDisplayMode(.large)
        }
        .sheet(isPresented: $showingShareSheet) {
            if let url = viewModel.shareURL() {
                ShareSheet(activityItems: [url])
            }
        }
        .sheet(isPresented: $showingFilePicker) {
            DocumentPicker(
                allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio, .aiff],
                onPick: { url in
                    viewModel.setCustomVoiceFile(url)
                }
            )
        }
        .sheet(isPresented: $showingSavedVoicePicker) {
            SavedVoicesSheet(
                voices: viewModel.savedVoices,
                onSelect: { voice in
                    viewModel.setSavedVoice(voice)
                    showingSavedVoicePicker = false
                },
                onDelete: { voice in
                    viewModel.deleteSavedVoice(voice)
                }
            )
        }
        .alert("Save Voice", isPresented: $showingExportAlert) {
            TextField("Voice Name", text: $exportVoiceName)
            Button("Cancel", role: .cancel) { }
            Button("Save") {
                viewModel.exportCurrentVoice(name: exportVoiceName)
            }
        } message: {
            Text("Enter a name for this voice. It will be saved for memory-efficient reuse.")
        }
        .onAppear {
            viewModel.loadSavedVoices()
        }
    }

    // MARK: - Voice Selection Section

    private var voiceSelectionSection: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Voice", systemImage: "waveform.circle.fill")
                    .font(.headline)
                    .foregroundStyle(.primary)

                Spacer()

                Picker("Voice", selection: $viewModel.selectedVoice) {
                    ForEach(PocketVoice.allCases) { voice in
                        Text(voice.displayName).tag(voice)
                    }
                }
                .pickerStyle(.menu)
                .tint(.accentColor)
            }

            // Custom voice file picker (Clone Voice)
            if viewModel.selectedVoice.isCustom {
                Divider()

                HStack {
                    Image(systemName: "waveform.badge.plus")
                        .foregroundStyle(.secondary)

                    if let fileName = viewModel.customVoiceFileName {
                        Text(fileName)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)

                        Spacer()

                        Button {
                            viewModel.clearCustomVoiceFile()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Select audio file to clone")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Spacer()
                    }

                    Button {
                        showingFilePicker = true
                    } label: {
                        Text(viewModel.customVoiceFileName == nil ? "Browse" : "Change")
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                }

                // Save voice button (export to embedding)
                if viewModel.canExportVoice {
                    HStack {
                        Image(systemName: "square.and.arrow.down")
                            .foregroundStyle(.blue)
                        Text("Save this voice for efficient reuse")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Save Voice") {
                            exportVoiceName = viewModel.customVoiceFileName?.replacingOccurrences(of: ".wav", with: "")
                                .replacingOccurrences(of: ".mp3", with: "") ?? "My Voice"
                            showingExportAlert = true
                        }
                        .font(.caption)
                        .fontWeight(.medium)
                    }
                }

                if viewModel.isExportingVoice {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text("Saving voice...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }

                if viewModel.needsVoiceFile {
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.orange)
                        Text("Select an audio file to clone the voice")
                            .font(.caption)
                            .foregroundStyle(.orange)
                        Spacer()
                    }
                }

                // Memory warning
                HStack {
                    Image(systemName: "memorychip")
                        .foregroundStyle(.secondary)
                    Text("Uses ~1.8GB RAM. Save voice for lower memory usage.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }

            // Saved voice picker
            if viewModel.selectedVoice.isSaved {
                Divider()

                HStack {
                    Image(systemName: "person.wave.2")
                        .foregroundStyle(.secondary)

                    if let fileName = viewModel.savedVoiceFileName {
                        Text(fileName)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)

                        Spacer()

                        Button {
                            viewModel.clearSavedVoice()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Select a saved voice")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Spacer()
                    }

                    Button {
                        viewModel.loadSavedVoices()
                        showingSavedVoicePicker = true
                    } label: {
                        Text(viewModel.savedVoiceFileName == nil ? "Choose" : "Change")
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                }

                if viewModel.needsSavedVoice {
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.orange)
                        Text("Select a saved voice or clone a new one first")
                            .font(.caption)
                            .foregroundStyle(.orange)
                        Spacer()
                    }
                }

                // Memory efficient note
                HStack {
                    Image(systemName: "leaf")
                        .foregroundStyle(.green)
                    Text("Memory efficient (~700MB)")
                        .font(.caption2)
                        .foregroundStyle(.green)
                    Spacer()
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Text Input Section

    private var textInputSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Text to Speak", systemImage: "text.alignleft")
                    .font(.headline)
                    .foregroundStyle(.primary)

                Spacer()

                if !viewModel.text.isEmpty {
                    Button("Clear") {
                        viewModel.clearText()
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }
            }

            TextEditor(text: $viewModel.text)
                .font(.body)
                .frame(minHeight: 150)
                .scrollContentBackground(.hidden)
                .disabled(viewModel.generationState.isBusy)

            HStack {
                Spacer()
                Text("\(viewModel.text.count) characters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Status Section

    private var statusSection: some View {
        HStack(spacing: 12) {
            statusIcon
                .font(.title2)

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.generationState.statusText)
                    .font(.subheadline)
                    .fontWeight(.medium)

                // Show stats when completed
                if !viewModel.generationState.isBusy {
                    // Generation stats (when last op was not streaming)
                    if !viewModel.lastOperationWasStreaming,
                       let url = viewModel.audioURL {
                        Text(url.lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)

                        if let duration = viewModel.audioDuration,
                           let genTime = viewModel.generationTime,
                           let rtf = viewModel.realTimeFactor {
                            if let initTime = viewModel.initializationTime, viewModel.isFirstGeneration == false {
                                Text("Init: \(String(format: "%.1f", initTime))s • \(String(format: "%.1f", duration))s audio • \(String(format: "%.2f", genTime))s gen • \(String(format: "%.2f", rtf))x RTFx")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("\(String(format: "%.1f", duration))s audio • \(String(format: "%.2f", genTime))s gen • \(String(format: "%.2f", rtf))x RTFx")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    // Streaming stats (when last op was streaming)
                    if viewModel.lastOperationWasStreaming,
                       let chunkCount = viewModel.streamingChunkCount,
                       let totalTime = viewModel.streamingTotalTime {
                        if let duration = viewModel.audioDuration,
                           let firstChunk = viewModel.streamingFirstChunkTime,
                           let rtf = viewModel.streamingRTF {
                            if let initTime = viewModel.initializationTime {
                                Text("Init: \(String(format: "%.1f", initTime))s • First audio: \(String(format: "%.2f", firstChunk))s • \(String(format: "%.1f", duration))s audio • \(String(format: "%.2f", rtf))x RTFx")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("First audio: \(String(format: "%.2f", firstChunk))s • \(chunkCount) chunks • \(String(format: "%.2f", totalTime))s total • \(String(format: "%.2f", rtf))x RTFx")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            Spacer()

            if viewModel.generationState.isBusy {
                ProgressView()
                    .scaleEffect(0.8)
            }
        }
        .padding(16)
        .background(statusBackgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch viewModel.generationState {
        case .idle:
            Image(systemName: "circle.dashed")
                .foregroundStyle(.secondary)
        case .initializing:
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(.blue)
        case .generating:
            Image(systemName: "waveform")
                .foregroundStyle(.blue)
                .symbolEffect(.variableColor.iterative)
        case .streaming:
            Image(systemName: "antenna.radiowaves.left.and.right")
                .foregroundStyle(.orange)
                .symbolEffect(.variableColor.iterative)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
        }
    }

    private var statusBackgroundColor: Color {
        switch viewModel.generationState {
        case .error:
            return Color.red.opacity(0.1)
        case .completed:
            return Color.green.opacity(0.1)
        case .generating, .initializing:
            return Color.blue.opacity(0.1)
        case .streaming:
            return Color.orange.opacity(0.1)
        default:
            return Color(.secondarySystemGroupedBackground)
        }
    }

    // MARK: - Action Buttons Section

    private var actionButtonsSection: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(spacing: 8) {
                // Generate button
                ActionButton(
                    title: viewModel.generateButtonTitle,
                    icon: viewModel.generationState.isGenerating ? "stop.fill" : "waveform",
                    style: viewModel.generationState.isGenerating ? .destructive : .primary,
                    isEnabled: viewModel.canGenerate || viewModel.generationState.isGenerating
                ) {
                    viewModel.generateOrStop()
                }

                // Stream button
                ActionButton(
                    title: viewModel.streamButtonTitle,
                    icon: viewModel.generationState.isStreaming ? "stop.fill" : "antenna.radiowaves.left.and.right",
                    style: viewModel.generationState.isStreaming ? .destructive : .streaming,
                    isEnabled: viewModel.canStream || viewModel.generationState.isStreaming
                ) {
                    viewModel.streamOrStop()
                }

                // Play button
                ActionButton(
                    title: viewModel.isPlaying ? "Stop" : "Play",
                    icon: viewModel.isPlaying ? "stop.fill" : "play.fill",
                    style: .secondary,
                    isEnabled: viewModel.canPlay
                ) {
                    if viewModel.isPlaying {
                        viewModel.stopPlayback()
                    } else {
                        viewModel.play()
                    }
                }

                // Share button
                ActionButton(
                    title: "Share",
                    icon: "square.and.arrow.up",
                    style: .secondary,
                    isEnabled: viewModel.canShare
                ) {
                    showingShareSheet = true
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
        }
    }
}

// MARK: - Action Button

struct ActionButton: View {
    enum Style {
        case primary
        case streaming
        case secondary
        case destructive
    }

    let title: String
    let icon: String
    let style: Style
    let isEnabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(iconColor)
                Text(title)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(textColor)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(backgroundColor)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1.0 : 0.5)
    }

    private var backgroundColor: Color {
        .clear
    }

    private var iconColor: Color {
        switch style {
        case .primary:
            return .accentColor
        case .streaming:
            return .orange
        case .secondary:
            return .primary
        case .destructive:
            return .red
        }
    }

    private var textColor: Color {
        .primary
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Document Picker

struct DocumentPicker: UIViewControllerRepresentable {
    let allowedContentTypes: [UTType]
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: allowedContentTypes)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void

        init(onPick: @escaping (URL) -> Void) {
            self.onPick = onPick
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }

            // Start accessing the security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                return
            }

            // Copy file to app's documents directory for persistent access
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let destinationURL = documentsPath.appendingPathComponent("custom_voice_\(UUID().uuidString).\(url.pathExtension)")

            do {
                // Remove existing file if present
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.copyItem(at: url, to: destinationURL)
                onPick(destinationURL)
            } catch {
                print("Failed to copy audio file: \(error)")
            }

            url.stopAccessingSecurityScopedResource()
        }
    }
}

// MARK: - Saved Voices Sheet

struct SavedVoicesSheet: View {
    let voices: [SavedVoice]
    let onSelect: (SavedVoice) -> Void
    let onDelete: (SavedVoice) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if voices.isEmpty {
                    emptyState
                } else {
                    voiceList
                }
            }
            .navigationTitle("Saved Voices")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text("No Saved Voices")
                .font(.headline)

            Text("Clone a voice using 'Custom Voice' and tap 'Save Voice' to create a memory-efficient voice embedding.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var voiceList: some View {
        List {
            Section {
                ForEach(voices) { voice in
                    VoiceRow(voice: voice) {
                        onSelect(voice)
                    }
                }
                .onDelete { indexSet in
                    for index in indexSet {
                        onDelete(voices[index])
                    }
                }
            } header: {
                Text("Tap to select a voice")
            } footer: {
                Text("Saved voices use ~700MB RAM compared to ~1.8GB for live cloning.")
            }
        }
    }
}

// MARK: - Voice Row

struct VoiceRow: View {
    let voice: SavedVoice
    let onSelect: () -> Void

    private var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: voice.createdAt)
    }

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 12) {
                Image(systemName: "person.wave.2.fill")
                    .font(.title2)
                    .foregroundStyle(.blue)
                    .frame(width: 40, height: 40)
                    .background(Color.blue.opacity(0.1))
                    .clipShape(Circle())

                VStack(alignment: .leading, spacing: 2) {
                    Text(voice.name)
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)

                    Text(formattedDate)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Image(systemName: "checkmark.circle")
                    .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Keyboard Dismissal

extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

#Preview {
    ContentView()
}
