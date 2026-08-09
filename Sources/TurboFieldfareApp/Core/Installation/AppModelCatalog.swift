import Foundation

/// One installable model the app knows how to fetch and run.
///
/// Wraps an `AppModelInstallDescriptor` (repo/revision/checksum/size, the
/// data `RepackModelInstallerClient` actually needs) with catalog-only
/// display metadata and a stable `id` used to key its install directory
/// (`AppModelLocation.directoryName(forCatalogID:)`) and its sibling
/// settings file (`MacAppSettingsFileStore`, already keyed by directory).
public struct AppModelCatalogEntry: Equatable, Identifiable, Sendable {
    public let id: String
    public let descriptor: AppModelInstallDescriptor
    public let summary: String
    public let minimumRecommendedRAMBytes: UInt64

    public init(id: String,
                descriptor: AppModelInstallDescriptor,
                summary: String,
                minimumRecommendedRAMBytes: UInt64) {
        self.id = id
        self.descriptor = descriptor
        self.summary = summary
        self.minimumRecommendedRAMBytes = minimumRecommendedRAMBytes
    }
}

/// The curated list of models the app can install and switch between.
///
/// This is a hand-maintained catalog, not a user-editable list of arbitrary
/// repo IDs — every entry needs a pinned revision and checksum the same way
/// `AppModelInstallDescriptor.default` already required.
public enum AppModelCatalog {
    public static let gemma4 = AppModelCatalogEntry(
        id: "gemma4",
        descriptor: AppModelInstallDescriptor.default,
        summary: "Gemma 4 26B-A4B IT, 4-bit — general-purpose instruct model.",
        minimumRecommendedRAMBytes: 16 * 1_024 * 1_024 * 1_024)

    /// The full catalog, in display order. Extend with additional entries
    /// (e.g. Qwen3.6-35B-A3B) only once that model's own repack/install path
    /// has been verified end-to-end with a real pinned repo/revision/SHA —
    /// see PHASE-LOG.md's Phase 3-6b history for that model's bring-up state.
    public static let entries: [AppModelCatalogEntry] = [gemma4]

    public static func entry(withID id: String) -> AppModelCatalogEntry? {
        entries.first { $0.id == id }
    }
}
