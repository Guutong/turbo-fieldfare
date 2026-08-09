import Testing
import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite struct ServerContextCapTests {
    @Test func defaultContextIsTheSharedCap() throws {
        let arguments = try ServerArguments.parse(["--model", "m.gturbo"])
        #expect(arguments.maxContext == ContextCap.maximum)
    }

    @Test func allowedValuesStayWithinTheCap() {
        #expect(ContextCap.allowedServerValues.allSatisfy { $0 <= ContextCap.maximum })
        #expect(ContextCap.allowedServerValues.contains(ContextCap.maximum))
    }

    @Test func oversizedContextIsRejectedWithAClearMessage() {
        do {
            _ = try ServerArguments.parse([
                "--model", "m.gturbo", "--max-context", "32768",
            ])
            Issue.record("expected 32768 to be rejected by the context cap")
        } catch let error as ServerArgumentError {
            let text = "\(error)"
            #expect(text.contains("\(ContextCap.maximum)"))
            #expect(text.contains("32768"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}
