import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// Lists the app's model catalog and lets the user switch to an
/// already-installed entry. Installing a *different, not-yet-installed*
/// catalog entry needs `AppModel`'s installer to be re-pointed at that
/// entry's descriptor, which isn't wired up yet (deferred until a second
/// real catalog entry — e.g. Qwen3.6, once its own bring-up is complete —
/// gives that plumbing something real to install). Today's single entry
/// (Gemma 4) already goes through the existing `ModelInstallView` flow the
/// first time the app runs, so this view's job is picking among what's
/// already on disk, not driving new installs.
struct ModelPickerView: View {
    let model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(AppModelCatalog.entries) { entry in
                row(for: entry)
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .frame(minWidth: 420, minHeight: 280)
    }

    private func row(for entry: AppModelCatalogEntry) -> some View {
        let isCurrent = model.modelPathText == currentPath(for: entry)
        return Button {
            select(entry)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(entry.descriptor.displayName)
                            .font(.body.weight(.medium))
                        if isCurrent {
                            Text("Current")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                        }
                    }
                    Text(entry.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(statusLabel(for: entry))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Text(MetricFormat.storage(entry.descriptor.approximateDownloadBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isCurrent || !isInstalled(entry) || model.isRunning)
    }

    private func statusLabel(for entry: AppModelCatalogEntry) -> String {
        isInstalled(entry) ? "Installed" : "Not installed — install from the model screen first"
    }

    private func isInstalled(_ entry: AppModelCatalogEntry) -> Bool {
        AppModelInstallationProbe.status(
            at: URL(fileURLWithPath: currentPath(for: entry)),
            descriptor: entry.descriptor) == .complete
    }

    private func currentPath(for entry: AppModelCatalogEntry) -> String {
        AppModelLocation.defaultURL(forCatalogID: entry.id).path
    }

    private func select(_ entry: AppModelCatalogEntry) {
        model.setModelURL(URL(fileURLWithPath: currentPath(for: entry)))
        dismiss()
    }
}
