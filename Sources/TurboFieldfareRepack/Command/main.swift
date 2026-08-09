import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--model <id>] [--overwrite] [--resume]
  TurboFieldfareRepack --output <model.gturbo> --local-checkpoint <path>
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams a supported checkpoint from Hugging Face and repackages
it without materializing the source checkpoint on disk. --model selects which
catalog entry to install (default: \(SupportedModelSource.default.id)); valid ids:
\(SupportedModelSource.all.map(\.id).joined(separator: ", ")). Set HF_TOKEN only
if Hugging Face requests authentication.

--local-checkpoint reads from a local HF-style checkpoint directory instead of
streaming from Hugging Face. The directory must contain config.json,
model.safetensors.index.json, and the safetensors shards.

A cancelled or interrupted download can be continued with --resume or removed
with --discard-partial (remote mode only; local checkpoints do not support
resume).
"""

private struct Arguments {
    var output: String?
    var model: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var localCheckpoint: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--output", "--input-gturbo", "--model", "--local-checkpoint":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                switch flag {
                case "--output":            parsed.output = values[index + 1]
                case "--input-gturbo":      parsed.inputGTurbo = values[index + 1]
                case "--model":             parsed.model = values[index + 1]
                case "--local-checkpoint":  parsed.localCheckpoint = values[index + 1]
                default: break
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        guard !(parsed.resume && parsed.localCheckpoint != nil) else {
            throw ParseError.invalidMode("--resume and --local-checkpoint are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall,
                  parsed.model == nil else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  parsed.model == nil else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }

    let options: RemoteStreamingRepackOptions
    if let localPath = arguments.localCheckpoint {
        // Local checkpoint mode: reads all files from a local directory.
        // repoID/revision are synthetic — they exist only for checkpoint
        // identity and are not used for network requests.
        let repoID = "local/checkpoint"
        let revision = "local"
        options = RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: output,
            token: nil,
            requireKnownSource: false,
            overwrite: arguments.overwrite,
            resume: arguments.resume,
            localCheckpointPath: localPath)
    } else {
        let source: ModelSource
        do {
            source = try SupportedModelSource.resolve(modelID: arguments.model)
        } catch ModelSelectionError.unknownID(let id, let validIDs) {
            printError("error: unknown model id \"\(id)\"; valid ids: "
                + "\(validIDs.joined(separator: ", "))\n\n\(usage)")
            return 2
        } catch {
            printError("install failed: \(error)")
            return 1
        }
        options = source.installOptions(
            outputDirectory: URL(fileURLWithPath: output),
            overwrite: arguments.overwrite,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            resume: arguments.resume)
    }
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        if arguments.localCheckpoint != nil {
            print("Repacked from local checkpoint")
        } else {
            let src = try SupportedModelSource.resolve(modelID: arguments.model)
            print("Installed \(src.displayName)")
        }
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
