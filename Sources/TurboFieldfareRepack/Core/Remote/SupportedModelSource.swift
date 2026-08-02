import Foundation

/// Failure modes for `SupportedModelSource.resolve(modelID:)`.
public enum ModelSelectionError: Error, CustomStringConvertible, Equatable {
    case unknownID(String, validIDs: [String])
    case notInstallable(String, reason: String)

    public var description: String {
        switch self {
        case .unknownID(let id, let validIDs):
            return "unknown model id \"\(id)\"; valid ids: \(validIDs.joined(separator: ", "))"
        case .notInstallable(let id, let reason):
            return "model \"\(id)\" is not installable: \(reason)"
        }
    }
}

/// One installable checkpoint: where it comes from, how big it is, and the
/// fingerprint that proves we fetched the revision we meant to.
public struct ModelSource: Equatable, Sendable, Identifiable {
    /// Stable key used for install directory names and persisted selection.
    /// Changing it orphans existing installs, so treat it as permanent.
    public let id: String
    public let displayName: String
    /// Compact label for space-constrained UI such as the status badge.
    public let shortDisplayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64
    /// False while the runtime still lacks kernels for this architecture. The
    /// catalog lists such models so the gap is visible in one place rather
    /// than implied by absence; callers must not offer them for install.
    public let isInstallable: Bool
    /// User-facing reason `isInstallable` is false. Nil when installable.
    /// Kept on the source itself so UI that lists the catalog doesn't
    /// restate why an entry is blocked in a second place.
    public let installBlockedReason: String?

    public init(id: String,
                displayName: String,
                shortDisplayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64 = 1_073_741_824,
                isInstallable: Bool,
                installBlockedReason: String? = nil) {
        self.id = id
        self.displayName = displayName
        self.shortDisplayName = shortDisplayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
        self.isInstallable = isInstallable
        self.installBlockedReason = installBlockedReason
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}

public enum SupportedModelSource {

    public static let gemma4_26B_A4B = ModelSource(
        id: "gemma4-26b-a4b",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        shortDisplayName: "Gemma 4 26B",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        isInstallable: true)

    /// poolside/Laguna-S-2.1 via the mlx-community mixed-precision conversion.
    ///
    /// Not installable yet. Group-128 routed experts decode and the repack
    /// planner understands this family's tensor naming, but this checkpoint
    /// quantizes **per layer**, not per role: its attention projections are
    /// 5-bit on 20 layers and 8-bit on the other 28. `ManifestQuantSlot.perLayer`
    /// can now express that, and `validateQuant` checks every layer's width, so
    /// this is back to being a missing-kernel problem: nothing decodes 5-bit
    /// attention. Per-head attention gating (`g_proj`) and YaRoP scaling are
    /// also parsed-but-unwired. See HANDOFF.md.
    ///
    /// The upstream revision is `d785a9349850807a34ac0ac1c22c66b718e77881`
    /// (48 layers, 256 experts, top-10, ~64.1 GB across 13 shards), recorded
    /// here rather than in `revision` on purpose: both `revision` and
    /// `sourceIndexSHA256` stay empty until this is genuinely installable,
    /// because an empty fingerprint makes the wrong-revision check pass
    /// vacuously. Fill both in together, in the change that flips
    /// `isInstallable`, never before.
    public static let lagunaS2_1 = ModelSource(
        id: "laguna-s-2-1",
        displayName: "Laguna-S-2.1 4-bit",
        shortDisplayName: "Laguna-S 2.1",
        repoID: "mlx-community/Laguna-S-2.1-oQ4e",
        revision: "",
        sourceIndexSHA256: "",
        approximateDownloadBytes: 64_130_000_000,
        installedBytes: 64_130_000_000,
        isInstallable: false,
        installBlockedReason:
            "Not yet runnable: the routed-expert path handles this checkpoint's "
            + "group-128 4-bit weights and the manifest can now describe its "
            + "per-layer quantization, but no attention kernel decodes the "
            + "5-bit weights it uses on 20 layers. Per-head attention gating "
            + "and YaRoP scaling are also unwired, and no revision or source "
            + "fingerprint has been pinned.")

    public static let all: [ModelSource] = [gemma4_26B_A4B, lagunaS2_1]

    public static var installable: [ModelSource] {
        all.filter(\.isInstallable)
    }

    public static func source(id: String) -> ModelSource? {
        all.first { $0.id == id }
    }

    /// The model used when nothing else is selected.
    public static let `default` = gemma4_26B_A4B

    /// Resolve a catalog entry from an optional `--model` id, applying the
    /// installer's selection rules in one place so the CLI stays thin:
    /// nil keeps the pinned default (no behavior change for existing
    /// callers), an unrecognized id is rejected with the valid set, and a
    /// cataloged-but-not-yet-installable entry is rejected with its own
    /// blocked reason rather than silently substituting the default.
    public static func resolve(modelID: String?) throws -> ModelSource {
        guard let modelID else { return `default` }
        guard let found = source(id: modelID) else {
            throw ModelSelectionError.unknownID(modelID, validIDs: all.map(\.id))
        }
        guard found.isInstallable else {
            throw ModelSelectionError.notInstallable(
                modelID, reason: found.installBlockedReason ?? "not installable")
        }
        return found
    }

    // MARK: - Compatibility surface
    //
    // Retained so the repack CLI and fingerprint table keep addressing the
    // single-model world without change while the catalog grows.

    public static var displayName: String { `default`.displayName }
    public static var repoID: String { `default`.repoID }
    public static var revision: String { `default`.revision }
    public static var sourceIndexSHA256: String { `default`.sourceIndexSHA256 }
    public static var approximateDownloadBytes: UInt64 { `default`.approximateDownloadBytes }
    public static var installedBytes: UInt64 { `default`.installedBytes }
    public static var reserveBytes: UInt64 { `default`.reserveBytes }

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        `default`.installOptions(outputDirectory: outputDirectory,
                                 overwrite: overwrite,
                                 token: token,
                                 resume: resume)
    }
}
