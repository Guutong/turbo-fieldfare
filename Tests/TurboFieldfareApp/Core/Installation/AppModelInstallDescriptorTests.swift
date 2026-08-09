import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareRepackCore

@Suite struct AppModelInstallDescriptorTests {
    /// `init(source:)` must track whatever `ModelSource` says rather than
    /// duplicating its own copy of the identity fields — otherwise a
    /// revision bump in the catalog would silently leave the app pinning
    /// the previous one, which is exactly the bug this migration fixes.
    @Test func descriptorTracksSourceFieldsRatherThanDuplicatingThem() {
        let source = ModelSource(
            id: "fixture-model",
            displayName: "Fixture Model",
            shortDisplayName: "Fixture",
            repoID: "example/fixture",
            revision: "deadbeef",
            sourceIndexSHA256: String(repeating: "a", count: 64),
            approximateDownloadBytes: 123,
            installedBytes: 456,
            reserveBytes: 789,
            isInstallable: true)

        let descriptor = AppModelInstallDescriptor(source: source, rangeStagingBytes: 10)

        #expect(descriptor.displayName == source.displayName)
        #expect(descriptor.shortDisplayName == source.shortDisplayName)
        #expect(descriptor.repoID == source.repoID)
        #expect(descriptor.revision == source.revision)
        #expect(descriptor.sourceIndexSHA256 == source.sourceIndexSHA256)
        #expect(descriptor.approximateDownloadBytes == source.approximateDownloadBytes)
        #expect(descriptor.installedBytes == source.installedBytes)
        #expect(descriptor.reserveBytes == source.reserveBytes)
        #expect(descriptor.rangeStagingBytes == 10)
    }

    @Test func shortDisplayNameFallsBackToDisplayNameWhenOmitted() {
        let descriptor = AppModelInstallDescriptor(
            displayName: "Fixture Model",
            repoID: "example/fixture",
            revision: "deadbeef",
            sourceIndexSHA256: String(repeating: "a", count: 64),
            approximateDownloadBytes: 1,
            installedBytes: 1,
            rangeStagingBytes: 1,
            reserveBytes: 1)
        #expect(descriptor.shortDisplayName == "Fixture Model")
    }

    @Test func defaultDescriptorMatchesCatalogDefault() {
        let descriptor = AppModelInstallDescriptor.default
        let source = SupportedModelSource.default
        #expect(descriptor.displayName == source.displayName)
        #expect(descriptor.repoID == source.repoID)
        #expect(descriptor.revision == source.revision)
        #expect(descriptor.sourceIndexSHA256 == source.sourceIndexSHA256)
    }

    /// A model the runtime can't decode yet must never surface as an
    /// installable descriptor. Nothing in the app should build one from a
    /// non-installable source.
    @Test func nonInstallableSourceIsExcludedFromCatalogInstallOffers() {
        #expect(!SupportedModelSource.installable.contains(SupportedModelSource.lagunaS2_1))
    }

    @Test func unavailableCatalogEntriesSurfacesNonInstallableSourcesWithAReason() {
        let entries = AppModelInstallDescriptor.unavailableCatalogEntries
        #expect(entries.contains { $0.id == SupportedModelSource.lagunaS2_1.id })
        #expect(!entries.contains { $0.id == SupportedModelSource.gemma4_26B_A4B.id })
        let laguna = entries.first { $0.id == SupportedModelSource.lagunaS2_1.id }
        #expect(laguna?.reason == SupportedModelSource.lagunaS2_1.installBlockedReason)
        #expect(laguna?.reason != nil)
    }
}
