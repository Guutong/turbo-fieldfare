import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    /// Built from the catalog's installable entries. A model that is listed
    /// but not yet installable contributes nothing, and an entry with an
    /// empty fingerprint is skipped outright — otherwise `modelID(for:)`
    /// would match an empty hash and report the wrong source as verified.
    public static let knownFingerprints: [String: String] = {
        var map: [String: String] = [:]
        for source in SupportedModelSource.installable
        where !source.sourceIndexSHA256.isEmpty {
            map[source.repoID] = source.sourceIndexSHA256
        }
        return map
    }()

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }
}
