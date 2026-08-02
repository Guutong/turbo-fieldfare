import Foundation
import TurboFieldfareRepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    /// Compact label for space-constrained UI. Defaults to `displayName`.
    public var shortDisplayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64

    public init(displayName: String,
                shortDisplayName: String? = nil,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.shortDisplayName = shortDisplayName ?? displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    /// Derived from the catalog rather than restated. These literals used to
    /// be duplicated here, so a revision bump in `SupportedModelSource` left
    /// the app pinning the previous one.
    public init(source: ModelSource,
                rangeStagingBytes: UInt64 = UInt64(RemoteChunkPolicy.defaultBytes)) {
        self.init(displayName: source.displayName,
                  shortDisplayName: source.shortDisplayName,
                  repoID: source.repoID,
                  revision: source.revision,
                  sourceIndexSHA256: source.sourceIndexSHA256,
                  approximateDownloadBytes: source.approximateDownloadBytes,
                  installedBytes: source.installedBytes,
                  rangeStagingBytes: rangeStagingBytes,
                  reserveBytes: source.reserveBytes)
    }

    public static let `default` = AppModelInstallDescriptor(
        source: SupportedModelSource.default)
}

/// A catalog entry the runtime can't install yet, plus why. Exposed so UI
/// can make the gap legible without depending on `TurboFieldfareRepackCore`
/// types directly — the Mac target only links `TurboFieldfareAppCore`.
public struct AppUnavailableModelInfo: Equatable, Sendable, Identifiable {
    public let id: String
    public let displayName: String
    public let reason: String?
}

extension AppModelInstallDescriptor {
    /// Catalog entries with `isInstallable == false`, for surfacing what's
    /// coming without offering it for install.
    public static var unavailableCatalogEntries: [AppUnavailableModelInfo] {
        SupportedModelSource.all
            .filter { !$0.isInstallable }
            .map {
                AppUnavailableModelInfo(id: $0.id,
                                        displayName: $0.displayName,
                                        reason: $0.installBlockedReason)
            }
    }
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
