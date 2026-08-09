import TurboFieldfareAppCore
import SwiftUI

/// Lets the user search HuggingFace Hub's public model listing and see
/// which results (if any) are already installable through this app's
/// catalog. Browse-only: this view never downloads raw HF weights — the
/// app has no pipeline for that (see `AppModelInstallDescriptor`'s doc
/// comment). Tapping a catalog-matched result routes into the existing
/// `ModelPickerView` install/select flow rather than reimplementing it.
struct HuggingFaceSearchView: View {
    let model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var results: [HuggingFaceModelSummary] = []
    @State private var isSearching = false
    @State private var errorText: String?
    @State private var searchTask: Task<Void, Never>?
    @State private var showingModelPicker = false

    private let client = HuggingFaceModelSearchClient()
    private static let debounceNanoseconds: UInt64 = 300_000_000

    var body: some View {
        NavigationStack {
            List {
                if let errorText {
                    Text(errorText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if results.isEmpty && !query.trimmingCharacters(in: .whitespaces).isEmpty && !isSearching {
                    Text("No models found.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                ForEach(results) { result in
                    row(for: result)
                }
            }
            .navigationTitle("Search HuggingFace")
            .searchable(text: $query, placement: .toolbar, prompt: "Search models…")
            .onChange(of: query) { _, newValue in
                scheduleSearch(for: newValue)
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .frame(minWidth: 480, minHeight: 360)
        .sheet(isPresented: $showingModelPicker) {
            ModelPickerView(model: model)
        }
    }

    private func row(for result: HuggingFaceModelSummary) -> some View {
        let catalogEntry = AppModelCatalog.entries.first { $0.descriptor.repoID == result.repoID }
        return Button {
            guard catalogEntry != nil else { return }
            showingModelPicker = true
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(result.repoID)
                        .font(.body.weight(.medium))
                    HStack(spacing: 8) {
                        Label(MetricFormat.count(result.downloads), systemImage: "arrow.down.circle")
                        Label(MetricFormat.count(result.likes), systemImage: "heart")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    if catalogEntry == nil {
                        Text("Not yet available as a TurboFieldfare model")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(catalogEntry == nil)
    }

    private func scheduleSearch(for newValue: String) {
        searchTask?.cancel()
        let trimmed = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            results = []
            errorText = nil
            isSearching = false
            return
        }
        searchTask = Task {
            try? await Task.sleep(nanoseconds: Self.debounceNanoseconds)
            guard !Task.isCancelled else { return }
            isSearching = true
            errorText = nil
            do {
                let found = try await client.search(query: trimmed)
                guard !Task.isCancelled else { return }
                results = found
            } catch {
                guard !Task.isCancelled else { return }
                errorText = "Search failed: \(error)"
            }
            isSearching = false
        }
    }
}
