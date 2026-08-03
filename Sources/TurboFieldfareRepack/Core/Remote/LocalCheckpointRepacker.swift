import Foundation

/// Repacks from a local HF-style checkpoint directory instead of streaming
/// from Hugging Face. Mirrors the remote flow (`RemoteSnapshotLoader.load` +
/// `HTTPRangeSourceByteProvider`) using filesystem reads.
enum LocalCheckpointRepacker {

    // MARK: - Snapshot loading

    /// Counterpart to `RemoteSnapshotLoader.load`. Reads metadata and shard
    /// headers from `checkpointPath` — a directory containing `config.json`,
    /// `model.safetensors.index.json`, and `model-0000N-of-0000M.safetensors`
    /// shards. The metadata files are copied into `metadataDirectory` so
    /// `IndexLoader.load` (which expects a directory of flat files) can parse
    /// them unchanged, and shard headers are parsed directly from the local
    /// safetensors files.
    static func loadSnapshot(checkpointPath: String,
                             metadataDirectory: String) throws -> RemoteSnapshot {
        try Posix.mkdirP(metadataDirectory)

        let srcIndex = (checkpointPath as NSString).appendingPathComponent("model.safetensors.index.json")
        let srcConfig = (checkpointPath as NSString).appendingPathComponent("config.json")
        let dstIndex = (metadataDirectory as NSString).appendingPathComponent("model.safetensors.index.json")
        let dstConfig = (metadataDirectory as NSString).appendingPathComponent("config.json")

        // Copy metadata files — identical content, no HTTP needed.
        try copyLocalFile(from: srcIndex, to: dstIndex)
        try copyLocalFile(from: srcConfig, to: dstConfig)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        let arch = try ArchInfo.load(configPath: dstConfig)

        // Build RemoteFileInfo for each shard by stat-ing the local file.
        let dummyCommit = "local-checkpoint-\(checkpointPath.hashValue)"
        var files: [String: RemoteFileInfo] = [:]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)

        for shard in metadata.shardFilenames {
            let localPath = (checkpointPath as NSString).appendingPathComponent(shard)
            let size = try Posix.fileSize(localPath)
            let info = RemoteFileInfo(filename: shard,
                                      resolvedCommit: dummyCommit,
                                      size: size,
                                      etag: nil,
                                      xetHash: nil,
                                      acceptsRanges: true)
            files[shard] = info

            // Parse the safetensors header from the local file (same 8-byte
            // prefix + header payload as RemoteSnapshotLoader, but via pread
            // instead of HTTP range request).
            let header = try Safetensors.parseLocalHeader(path: localPath,
                                                           fileSize: size,
                                                           filename: shard)
            headers.append(header)
        }

        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: dummyCommit,
                              metadataDirectory: metadataDirectory)
    }

    // MARK: - Helpers

    private static func copyLocalFile(from src: String, to dst: String) throws {
        try Posix.mkdirP((dst as NSString).deletingLastPathComponent)
        try? FileManager.default.removeItem(atPath: dst)
        try FileManager.default.copyItem(atPath: src, toPath: dst)
    }
}

// MARK: - Byte provider

/// `SourceByteProvider` that reads byte ranges from a local checkpoint
/// directory instead of HTTP. Same `copyBatch` structure as
/// `HTTPRangeSourceByteProvider` but replaces
/// `remote.downloadRangeToTempFile` with a local `pread`.
public final class LocalFileSourceByteProvider: SourceByteProvider {
    private let checkpointPath: String
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int

    public init(checkpointPath: String,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.checkpointPath = checkpointPath
        self.files = files
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }
        var copiedThisRun: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            guard let info = files[copy.shardID] else {
                throw RepackError.configurationInvalid(
                    detail: "missing file info for \(copy.shardID)")
            }
            if try Posix.entryKind(temporaryPath) != .absent {
                try FileManager.default.removeItem(atPath: temporaryPath)
            }

            // Read byte range from local shard file into a temp file.
            let shardPath = (checkpointPath as NSString).appendingPathComponent(copy.shardID)
            let rangeEnd = copy.sourceOffset + copy.size
            guard rangeEnd <= info.size else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "range \(copy.sourceOffset)-\(rangeEnd) exceeds \(copy.shardID) size \(info.size)")
            }
            try readLocalRangeToFile(
                path: shardPath,
                offset: copy.sourceOffset,
                length: Int(copy.size),
                targetPath: temporaryPath,
                scratch: scratch,
                audit: audit)
            copiedThisRun += copy.size
            progress(copiedThisRun)

            let sourceFD = try Posix.openReadNoFollow(temporaryPath)
            var touched = Set<String>()
            do {
                for destination in copy.destinations {
                    let destinationFD: Int32
                    if let existing = outputFDs[destination.destinationPath] {
                        destinationFD = existing
                    } else {
                        destinationFD = try Posix.openExistingRW(
                            destination.destinationPath)
                        outputFDs[destination.destinationPath] = destinationFD
                    }
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceFD: sourceFD,
                        sourcePath: temporaryPath,
                        destinationFD: destinationFD,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset - copy.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceFD)
            } catch {
                close(sourceFD)
                throw error
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            progress(copiedThisRun)
            try? FileManager.default.removeItem(atPath: temporaryPath)
            try Task.checkCancellation()
        }
    }

    /// Read `length` bytes at `offset` from `path` into a new file at `targetPath`.
    private func readLocalRangeToFile(
        path: String,
        offset: UInt64,
        length: Int,
        targetPath: String,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        if length == 0 {
            FileManager.default.createFile(atPath: targetPath, contents: Data())
            return
        }
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        // Create the output file, then loop pread→pwrite.
        FileManager.default.createFile(atPath: targetPath, contents: nil)
        let outFD = try Posix.openExistingRW(targetPath)
        defer { close(outFD) }
        var remaining = UInt64(length)
        var sourceOff = offset
        var destOff: UInt64 = 0
        while remaining > 0 {
            let chunk = min(Int(remaining), scratch.count)
            try Posix.preadAll(fd: fd, path: path,
                               buf: scratch.baseAddress!,
                               count: chunk, offset: sourceOff)
            try Posix.pwriteAll(fd: outFD, path: targetPath,
                                buf: scratch.baseAddress!,
                                count: chunk, offset: destOff)
            audit.recordRead(bytes: chunk)
            audit.recordWrite(bytes: chunk)
            remaining -= UInt64(chunk)
            sourceOff += UInt64(chunk)
            destOff += UInt64(chunk)
        }
        try Posix.fsync(outFD, path: targetPath)
    }

    /// Direct port of `HTTPRangeSourceByteProvider.copyBytes`.
    private func copyBytes(
        sourceFD: Int32,
        sourcePath: String,
        destinationFD: Int32,
        destinationPath: String,
        sourceOffset: UInt64,
        destinationOffset: UInt64,
        size: UInt64,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}
