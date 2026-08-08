import Foundation
import TurboFieldfareFormat

public struct LocalRepackResult: Sendable {
    public let displayName: String
    public let resolvedCommit: String
    public let outputDir: String
}

public enum LocalRepacker {

    public static func repack(snapshotDir: String,
                               outputDirectory: String,
                               overwrite: Bool) throws -> LocalRepackResult {

        let indexPath = (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")

        guard FileManager.default.fileExists(atPath: indexPath) else {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "not found")
        }
        guard FileManager.default.fileExists(atPath: configPath) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not found")
        }

        if try Posix.entryKind(outputDirectory) == .directory, !overwrite {
            throw RepackError.configurationInvalid(
                detail: "output directory already exists: \(outputDirectory)")
        }

        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(configPath: configPath)

        var shardHeaders: [Safetensors.Header] = []
        shardHeaders.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let shardPath = (snapshotDir as NSString).appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: shardPath) else {
                throw RepackError.safetensorsHeaderInvalid(path: shardPath, detail: "not found")
            }
            let fileSize = try FileManager.default.attributesOfItem(atPath: shardPath)[.size] as? UInt64 ?? 0
            let fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: shardPath))
            defer { try? fileHandle.close() }
            guard let prefixData = try fileHandle.read(upToCount: 8), prefixData.count == 8 else {
                throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header prefix")
            }
            let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
                var value: UInt64 = 0
                for i in 0..<8 {
                    value |= UInt64(raw[i]) << UInt64(i * 8)
                }
                return value
            }
            if headerSize > Safetensors.maxHeaderBytes || headerSize > fileSize - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            guard let headerData = try fileHandle.read(upToCount: Int(headerSize)),
                  headerData.count == headerSize else {
                throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header body")
            }
            shardHeaders.append(try Safetensors.parseHeaderBytes(path: shard,
                                                                  fileSize: fileSize,
                                                                  headerBytes: headerData))
        }

        let outputURL = URL(fileURLWithPath: outputDirectory)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let plan = try RepackPlanner.plan(meta: metadata,
                                          arch: arch,
                                          shardHeaders: shardHeaders,
                                          outputDir: outputDirectory)

        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        _ = try DiskSpaceChecker.requireAvailable(
            path: outputDirectory,
            bytes: outputBytes + UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: 1 * 1024 * 1024 * 1024)

        // Create output files
        let residentPath = (outputDirectory as NSString).appendingPathComponent("model_weights.bin")
        FileManager.default.createFile(atPath: residentPath,
                                       contents: Data(),
                                       attributes: [.posixPermissions: 0o644])
        let residentFD = try Posix.openCreateRW(residentPath)
        defer { close(residentFD) }
        try Posix.ftruncate(residentFD, path: residentPath, size: plan.resident.totalSize)

        let layersDir = (outputDirectory as NSString).appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: URL(fileURLWithPath: layersDir),
                                                withIntermediateDirectories: true)

        for layerPlan in plan.layers {
            let layerPath = layerPlan.path
            FileManager.default.createFile(atPath: layerPath,
                                           contents: Data(),
                                           attributes: [.posixPermissions: 0o644])
            let layerFD = try Posix.openCreateRW(layerPath)
            defer { close(layerFD) }
            try Posix.ftruncate(layerFD, path: layerPath, size: layerPlan.fileSize)
        }

        // Write layout.json
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layout = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        let layoutPath = (layersDir as NSString).appendingPathComponent("layout.json")
        try layout.write(to: URL(fileURLWithPath: layoutPath), options: Data.WritingOptions.atomic)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)

        // Write manifest.json — every entry carries its real SHA-256 so
        // --verify-install (which re-hashes every declared file) passes.
        let audit = RepackAudit()
        func hashedEntry(path: String) throws -> GTurboJSON.FileEntry {
            let size = try FileManager.default.attributesOfItem(atPath: path)[.size] as? UInt64 ?? 0
            let sha = try WriterCore.hashEntireFile(path: path, size: size, audit: audit)
            return GTurboJSON.FileEntry(size: size, sha256: sha)
        }
        var files: [(relativePath: String, info: GTurboJSON.FileEntry)] = []
        files.append(("model_weights.bin", try hashedEntry(path: residentPath)))
        for layerPlan in plan.layers {
            let fileName = (layerPlan.path as NSString).lastPathComponent
            files.append(("packed_experts/\(fileName)", try hashedEntry(path: layerPlan.path)))
        }
        files.append(("packed_experts/layout.json", try hashedEntry(path: layoutPath)))

        let manifestData = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "qwen3.6-35b-a3b-4bit",
            sourceSnapshotHash: metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.arch.numExperts,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: metadata.baseBits,
                attention: metadata.baseBits,
                router: metadata.baseBits,
                sharedExpert: metadata.baseBits,
                routedExpert: metadata.baseBits))

        try manifestData.write(to: URL(fileURLWithPath: (outputDirectory as NSString).appendingPathComponent("manifest.json")),
                               options: Data.WritingOptions.atomic)

        return LocalRepackResult(displayName: "Qwen3.6-35B-A3B-4bit",
                                 resolvedCommit: metadata.indexSha256Hex,
                                 outputDir: outputDirectory)
    }
}
