import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelCatalogTests {
    @Test func catalogHasAtLeastOneEntry() {
        #expect(!AppModelCatalog.entries.isEmpty)
    }

    @Test func entryIDsAreUnique() {
        let ids = AppModelCatalog.entries.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func defaultDescriptorMatchesCatalogFirstEntry() {
        #expect(AppModelInstallDescriptor.default == AppModelCatalog.entries.first?.descriptor)
    }

    @Test func entryLookupByIDFindsKnownEntry() {
        #expect(AppModelCatalog.entry(withID: "gemma4") == AppModelCatalog.gemma4)
    }

    @Test func entryLookupByIDReturnsNilForUnknown() {
        #expect(AppModelCatalog.entry(withID: "does-not-exist") == nil)
    }

    @Test func gemma4CatalogDirectoryMatchesLegacyDefault() {
        let catalogPath = AppModelLocation.defaultURL(forCatalogID: "gemma4").path
        let legacyPath = AppModelLocation.defaultURL().path
        #expect(catalogPath == legacyPath)
    }
}
