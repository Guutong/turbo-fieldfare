import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct CLIContextCapTests {
    @Test func capBoundaryIsAccepted() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-context", String(ContextCap.maximum),
        ])
        #expect(arguments.maxContext == ContextCap.maximum)
    }

    @Test func oversizedContextIsRejectedWithAClearMessage() {
        let over = ContextCap.maximum + 1
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--max-context", String(over),
            ])
        }
        do {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--max-context", String(over),
            ])
            Issue.record("expected the context cap to reject \(over)")
        } catch let error as ArgsError {
            let text = error.description
            #expect(text.contains("\(ContextCap.maximum)"))
            #expect(text.contains("\(over)"))
            #expect(text.lowercased().contains("exceeds"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test func nonPositiveContextIsRejected() {
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--max-context", "0",
            ])
        }
    }

    /// The guard must fire before any model/tokenizer work, so a bogus model
    /// path still yields the context error (not a file-not-found crash).
    @Test func runRejectsOversizedContextBeforeTouchingTheModel() async throws {
        let pipe = Pipe()
        let args = Args(model: "/nonexistent/does-not-exist.gturbo",
                        prompt: "The capital of France is",
                        maxContext: ContextCap.maximum + 4_096)
        let result = await run(args: args,
                              stdout: FileHandle.nullDevice,
                              stderr: pipe.fileHandleForWriting)
        try pipe.fileHandleForWriting.close()
        let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
                          as: UTF8.self)
        #expect(result.exitCode == 2)
        #expect(text.contains("exceeds the maximum supported context"))
        #expect(text.contains("\(ContextCap.maximum)"))
    }

    @Test func promptRejectionReasonReportsBothCounts() {
        let reason = ContextCap.promptRejectionReason(promptTokens: 20_000,
                                                      maxContext: 16_384)
        #expect(reason?.contains("20000") == true)
        #expect(reason?.contains("16384") == true)
        #expect(ContextCap.promptRejectionReason(promptTokens: 16_383,
                                                 maxContext: 16_384) == nil)
        // A prompt exactly at the cap leaves no room to generate: still rejected.
        #expect(ContextCap.promptRejectionReason(promptTokens: 16_384,
                                                 maxContext: 16_384) != nil)
    }
}
