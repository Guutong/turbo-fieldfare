import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct SupportedModelSourceTests {
    @Test func installableIncludesSupportedCatalogEntries() {
        let installable = SupportedModelSource.installable
        #expect(installable.contains(SupportedModelSource.gemma4_26B_A4B))
        #expect(installable.contains(SupportedModelSource.lagunaS2_1))
        #expect(installable.allSatisfy { $0.isInstallable })
    }

    @Test func defaultIsInstallable() {
        #expect(SupportedModelSource.default.isInstallable)
    }

    @Test func sourceLookupResolvesByID() {
        #expect(SupportedModelSource.source(id: SupportedModelSource.gemma4_26B_A4B.id)
            == SupportedModelSource.gemma4_26B_A4B)
        #expect(SupportedModelSource.source(id: SupportedModelSource.lagunaS2_1.id)
            == SupportedModelSource.lagunaS2_1)
        #expect(SupportedModelSource.source(id: "unknown-model") == nil)
    }

    @Test func installableEntriesHavePinnedRevisionAndFingerprint() {
        #expect(SupportedModelSource.lagunaS2_1.isInstallable)
        #expect(!SupportedModelSource.lagunaS2_1.revision.isEmpty)
        #expect(!SupportedModelSource.lagunaS2_1.sourceIndexSHA256.isEmpty)
    }

    @Test func installableEntriesHaveNoBlockedReason() {
        #expect(SupportedModelSource.lagunaS2_1.installBlockedReason == nil)
        #expect(SupportedModelSource.gemma4_26B_A4B.installBlockedReason == nil)
    }

    // MARK: - resolve(modelID:)

    @Test func resolveWithNoIDKeepsTheGemmaDefault() throws {
        let resolved = try SupportedModelSource.resolve(modelID: nil)
        #expect(resolved == SupportedModelSource.default)
        #expect(resolved == SupportedModelSource.gemma4_26B_A4B)
    }

    @Test func resolveAcceptsValidInstallableIDs() throws {
        let gemma = try SupportedModelSource.resolve(modelID: "gemma4-26b-a4b")
        #expect(gemma == SupportedModelSource.gemma4_26B_A4B)
        let laguna = try SupportedModelSource.resolve(modelID: "laguna-s-2-1")
        #expect(laguna == SupportedModelSource.lagunaS2_1)
    }

    @Test func resolveRejectsAnUnknownID() {
        #expect(throws: ModelSelectionError.unknownID(
            "nonexistent-model",
            validIDs: SupportedModelSource.all.map(\.id))) {
            try SupportedModelSource.resolve(modelID: "nonexistent-model")
        }
    }
}
