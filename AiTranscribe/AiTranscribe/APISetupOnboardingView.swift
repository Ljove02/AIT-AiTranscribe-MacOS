import SwiftUI

struct APISetupOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void

    @AppStorage("selectedOpenAITranscriptionModel") private var selectedTranscriptionModel = TranscriptionModelDescriptor.openAIModels[0].id
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.openAI.id

    @State private var selectedProviderId = TranscriptionProviderDescriptor.openAI.id
    @State private var apiKey = ""
    @State private var savedProviderIds: Set<String> = []
    @State private var apiModels = APITranscriptionModelStore.load()
    @State private var errorMessage: String?

    private let recommendationDate = "July 1, 2026"

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingHeader(
                eyebrow: "API setup",
                title: "Add API providers",
                subtitle: "Save keys in Keychain. Recommendations updated \(recommendationDate)."
            )
            .padding(.bottom, 32)

            HStack(alignment: .top, spacing: 46) {
                providerColumn
                    .frame(width: 316, alignment: .topLeading)

                recommendationColumn
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }

            Spacer()

            OnboardingNavBar(
                nextTitle: apiModels.isEmpty ? "Skip API" : "Continue",
                nextIcon: "arrow.right",
                onBack: onBack,
                onNext: onNext
            )
        }
        .padding(34)
        .onAppear {
            refreshSavedProviders()
        }
        .onChange(of: selectedProviderId) { _, _ in
            apiKey = ""
            errorMessage = nil
            refreshSavedProviders()
        }
    }

    private var providerColumn: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Provider")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 5) {
                ForEach(apiProviders) { provider in
                    APIProviderChoiceRow(
                        provider: provider,
                        systemImage: providerIcon(provider.id),
                        isSelected: selectedProviderId == provider.id,
                        isSaved: savedProviderIds.contains(provider.id)
                    ) {
                        selectedProviderId = provider.id
                    }
                }
            }

            apiKeySection
                .padding(.top, 6)

            APIInlineNote(systemImage: "shield.lefthalf.filled", text: providerPrivacyText)
                .padding(.top, 4)
        }
    }

    private var apiKeySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("API key")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .tracking(0.9)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            SecureField(apiKeyPlaceholder, text: $apiKey)
                .textFieldStyle(.plain)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .padding(.horizontal, 12)
                .frame(height: 36)
                .background(.black.opacity(0.20), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                )
                .onSubmit(saveAPIKey)

            HStack(spacing: 10) {
                Button {
                    saveAPIKey()
                } label: {
                    Label(isProviderSaved ? "Key Saved" : "Save Key", systemImage: isProviderSaved ? "checkmark.circle.fill" : "key.fill")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)

                if isProviderSaved {
                    Text("Stored in macOS Keychain")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.green)
                }
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var recommendationColumn: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .center, spacing: 14) {
                Image(systemName: providerIcon(selectedProviderId))
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(.blue.opacity(0.88), in: Circle())
                    .shadow(color: .blue.opacity(0.22), radius: 14, x: 0, y: 8)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Recommended models")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    Text(selectedProvider.displayName)
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .tracking(1.0)
                        .textCase(.uppercase)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(recommendedModels) { model in
                    APIRecommendationCard(
                        model: model,
                        isConfigured: apiModels.contains(where: { $0.providerId == model.providerId && $0.modelId == model.modelId }),
                        action: { addRecommended(model) }
                    )
                }
            }

            if selectedProviderId == TranscriptionProviderDescriptor.openRouter.id {
                APIInlineNote(
                    systemImage: "plus.message.fill",
                    text: "Add custom OpenRouter speech-to-text slugs later in Settings > Models."
                )
                .padding(.top, 2)
            }
        }
    }

    private var apiProviders: [TranscriptionProviderDescriptor] {
        [.openAI, .elevenLabs, .openRouter]
    }

    private var selectedProvider: TranscriptionProviderDescriptor {
        apiProviders.first { $0.id == selectedProviderId } ?? .openAI
    }

    private var isProviderSaved: Bool {
        savedProviderIds.contains(selectedProviderId)
    }

    private var apiKeyPlaceholder: String {
        if isProviderSaved {
            return "••••••••••••••••"
        }
        return "Paste \(selectedProvider.displayName) API key"
    }

    private var providerPrivacyText: String {
        switch selectedProviderId {
        case TranscriptionProviderDescriptor.openAI.id:
            return "OpenAI models send audio to OpenAI. Check workspace retention for sensitive audio."
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "ElevenLabs Scribe sends audio to ElevenLabs for speech-to-text."
        case TranscriptionProviderDescriptor.openRouter.id:
            return "OpenRouter routes requests to third-party providers. Use speech-to-text slugs."
        default:
            return selectedProvider.privacyDescription
        }
    }

    private var recommendedModels: [APIOnboardingModelRecommendation] {
        switch selectedProviderId {
        case TranscriptionProviderDescriptor.openAI.id:
            return [
                APIOnboardingModelRecommendation(
                    providerId: TranscriptionProviderDescriptor.openAI.id,
                    providerName: TranscriptionProviderDescriptor.openAI.displayName,
                    modelId: "gpt-4o-transcribe",
                    displayName: "GPT-4o Transcribe",
                    note: "Quality baseline",
                    audioCostPerMinuteUSD: 0.006
                ),
                APIOnboardingModelRecommendation(
                    providerId: TranscriptionProviderDescriptor.openAI.id,
                    providerName: TranscriptionProviderDescriptor.openAI.displayName,
                    modelId: "gpt-4o-mini-transcribe-2025-12-15",
                    displayName: "GPT-4o Mini Transcribe",
                    note: "Fast, low-cost default",
                    audioCostPerMinuteUSD: 0.003
                )
            ]
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return [
                APIOnboardingModelRecommendation(
                    providerId: TranscriptionProviderDescriptor.elevenLabs.id,
                    providerName: TranscriptionProviderDescriptor.elevenLabs.displayName,
                    modelId: "scribe_v2",
                    displayName: "ElevenLabs Scribe v2",
                    note: "Recommended batch STT model",
                    audioCostPerMinuteUSD: nil
                )
            ]
        default:
            return [
                APIOnboardingModelRecommendation(
                    providerId: TranscriptionProviderDescriptor.openRouter.id,
                    providerName: TranscriptionProviderDescriptor.openRouter.displayName,
                    modelId: "openai/whisper-1",
                    displayName: "OpenRouter Whisper",
                    note: "Example STT slug",
                    audioCostPerMinuteUSD: nil
                )
            ]
        }
    }

    private func saveAPIKey() {
        do {
            try APIKeychainStore.saveAPIKey(apiKey, providerId: selectedProviderId)
            apiKey = ""
            errorMessage = nil
            refreshSavedProviders()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func refreshSavedProviders() {
        savedProviderIds = Set(apiProviders.filter { APIKeychainStore.hasAPIKey(providerId: $0.id) }.map(\.id))
    }

    private func addRecommended(_ recommendation: APIOnboardingModelRecommendation) {
        guard APITranscriptionModelStore.isSupportedModel(recommendation.modelId, providerId: recommendation.providerId) else {
            errorMessage = APITranscriptionModelStore.supportedModelHint(providerId: recommendation.providerId)
            return
        }

        let model = APITranscriptionModel(
            providerId: recommendation.providerId,
            providerName: recommendation.providerName,
            modelId: recommendation.modelId,
            displayName: recommendation.displayName,
            audioCostPerMinuteUSD: recommendation.audioCostPerMinuteUSD
        )
        apiModels = APITranscriptionModelStore.upsert(model)
        selectedTranscriptionProviderId = recommendation.providerId
        selectedTranscriptionModel = recommendation.modelId
        errorMessage = nil
    }

    private func providerIcon(_ providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openAI.id: return "sparkles"
        case TranscriptionProviderDescriptor.elevenLabs.id: return "waveform"
        case TranscriptionProviderDescriptor.openRouter.id: return "arrow.triangle.branch"
        default: return "globe"
        }
    }
}

private struct APIProviderChoiceRow: View {
    let provider: TranscriptionProviderDescriptor
    let systemImage: String
    let isSelected: Bool
    let isSaved: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: systemImage)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(isSelected ? .white : .secondary)
                    .frame(width: 34, height: 34)
                    .background(isSelected ? .blue : .white.opacity(0.08), in: Circle())

                VStack(alignment: .leading, spacing: 2) {
                    Text(provider.displayName)
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    Text(isSaved ? "Key saved" : "Add key")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(isSaved ? .green : .secondary)
                }

                Spacer()

                Image(systemName: isSelected ? "checkmark.circle.fill" : "arrow.right.circle")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(isSelected ? .blue : .secondary.opacity(0.68))
            }
            .padding(.vertical, 11)
            .padding(.horizontal, 12)
            .contentShape(.rect)
            .background(alignment: .leading) {
                Capsule()
                    .fill(Color.blue)
                    .frame(width: isSelected ? 4 : 0)
                    .padding(.vertical, 9)
            }
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(.white.opacity(0.07))
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(provider.displayName), \(isSaved ? "key saved" : "key not saved")")
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

private struct APIInlineNote: View {
    let systemImage: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.blue)
                .padding(.top, 1)

            Text(text)
                .font(.system(size: 12, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: 0)
        }
    }
}

private struct APIOnboardingModelRecommendation: Identifiable {
    var id: String { "\(providerId):\(modelId)" }
    let providerId: String
    let providerName: String
    let modelId: String
    let displayName: String
    let note: String
    let audioCostPerMinuteUSD: Double?
}

private struct APIRecommendationCard: View {
    let model: APIOnboardingModelRecommendation
    let isConfigured: Bool
    let action: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: isConfigured ? "checkmark.circle.fill" : "plus.circle.fill")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(isConfigured ? .green : .blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 5) {
                Text(model.displayName)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(model.modelId)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text(model.note)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if let cost = model.audioCostPerMinuteUSD {
                Text("$\(cost, specifier: "%.3f")/min")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .foregroundStyle(.green)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(.green.opacity(0.12), in: Capsule())
            }

            if isConfigured {
                Text("Added")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .frame(width: 68)
            } else {
                Button("Add") {
                    action()
                }
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .frame(width: 68)
            }
        }
        .padding(.vertical, 13)
        .padding(.horizontal, 14)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(isConfigured ? .green.opacity(0.055) : .white.opacity(0.035))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isConfigured ? .green.opacity(0.18) : .white.opacity(0.08), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    APISetupOnboardingView(onNext: {}, onBack: {})
        .frame(width: 900, height: 720)
}
