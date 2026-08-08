import Foundation
import Darwin

/// Writes a single routed-expert `packed_experts/layer_NN.bin` file: for
/// every logical expert, copies each of its per-expert sub-tensor slices
/// (gate/up/down x weights/scales/biases) from the source shard into the
/// expert's blob at `physicalRank(for:) * expertStride + offsetInExpertBlob`.
///
/// Mirrors `ResidentWriter.write`'s copy loop but over `LayerFilePlan`
/// instead of `ResidentFilePlan`. Without this, the file is left as the
/// zero-filled bytes `ftruncate` produces — silently "successful" but
/// numerically empty.
enum LayerWriter {

    static func write(layer: LayerFilePlan,
                       fd: Int32,
                       shardsByPath: inout [String: MmapHandle],
                       audit: RepackAudit) throws {
        guard layer.expertsPerLayer > 0 else { return }
        for expert in 0..<layer.expertsPerLayer {
            let blobBase = UInt64(layer.physicalRank(for: expert)) * layer.expertStride
            for slice in layer.subTensors {
                guard slice.sizeInExpertBlob > 0 else { continue }
                let shard = try mappedShard(path: slice.sourceTensor.shardPath,
                                            shardsByPath: &shardsByPath)
                let srcOffset = slice.sourceTensor.absoluteOffset
                    + UInt64(expert) * slice.sourceOffsetPerExpert
                let dstOffset = blobBase + slice.offsetInExpertBlob
                try WriterCore.pwriteTensorRegion(srcShard: shard,
                                                  srcAbsoluteOffset: srcOffset,
                                                  size: slice.sizeInExpertBlob,
                                                  dstFd: fd, dstPath: layer.path,
                                                  dstOffset: dstOffset,
                                                  audit: audit)
            }
        }
    }

    private static func mappedShard(path: String,
                                    shardsByPath: inout [String: MmapHandle]) throws -> MmapHandle {
        if let h = shardsByPath[path] { return h }
        let h = try MmapHandle(path: path)
        shardsByPath[path] = h
        return h
    }
}
