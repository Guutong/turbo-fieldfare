import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        #expect(mebibytes == [305, 385, 545])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, Default",
            "8K, +85 MB",
            "16K, +250 MB",
        ])
    }

    @Test func optionsNeverExceedTheSharedContextCap() {
        #expect(AppContextLengthOption.allCases.allSatisfy {
            ContextCap.allowedServerValues.contains($0.tokens)
        })
    }
}
