import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct SupportedModelSourceTests {
    @Test func installableExcludesEntriesThatAreNotYetInstallable() {
        let installable = SupportedModelSource.installable
        #expect(installable.contains(SupportedModelSource.gemma4_26B_A4B))
        #expect(!installable.contains(SupportedModelSource.lagunaS2_1))
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

    /// The doc comment on `lagunaS2_1` explains its empty revision and
    /// fingerprint are a deliberate guard against fetching an unpinned
    /// checkpoint — this pins that behavior so it can't regress silently.
    @Test func notYetInstallableEntryHasNoPinnedRevisionOrFingerprint() {
        #expect(!SupportedModelSource.lagunaS2_1.isInstallable)
        #expect(SupportedModelSource.lagunaS2_1.revision.isEmpty)
        #expect(SupportedModelSource.lagunaS2_1.sourceIndexSHA256.isEmpty)
    }

    @Test func notYetInstallableEntryExplainsWhy() {
        #expect(SupportedModelSource.lagunaS2_1.installBlockedReason != nil)
        #expect(SupportedModelSource.gemma4_26B_A4B.installBlockedReason == nil)
    }

    // MARK: - resolve(modelID:)

    @Test func resolveWithNoIDKeepsTheGemmaDefault() throws {
        let resolved = try SupportedModelSource.resolve(modelID: nil)
        #expect(resolved == SupportedModelSource.default)
        #expect(resolved == SupportedModelSource.gemma4_26B_A4B)
    }

    @Test func resolveAcceptsAValidInstallableID() throws {
        let resolved = try SupportedModelSource.resolve(modelID: "gemma4-26b-a4b")
        #expect(resolved == SupportedModelSource.gemma4_26B_A4B)
    }

    @Test func resolveRejectsAnUnknownID() {
        #expect(throws: ModelSelectionError.unknownID(
            "nonexistent-model",
            validIDs: SupportedModelSource.all.map(\.id))) {
            try SupportedModelSource.resolve(modelID: "nonexistent-model")
        }
    }

    @Test func resolveRejectsANotYetInstallableID() {
        #expect(throws: ModelSelectionError.notInstallable(
            "laguna-s-2-1",
            reason: SupportedModelSource.lagunaS2_1.installBlockedReason!)) {
            try SupportedModelSource.resolve(modelID: "laguna-s-2-1")
        }
    }
}
