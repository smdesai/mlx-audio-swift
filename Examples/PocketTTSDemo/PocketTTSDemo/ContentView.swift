//
//  ContentView.swift
//  PocketTTSDemo
//
//  Main UI for PocketTTS demo app.
//

import SwiftUI

struct ContentView: View {
    @State private var viewModel = TTSViewModel()
    @State private var showingShareSheet = false

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
    }

    // MARK: - Voice Selection Section

    private var voiceSelectionSection: some View {
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

// MARK: - Keyboard Dismissal

extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

#Preview {
    ContentView()
}
