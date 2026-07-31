import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `dequant_int5_gemv_simd` kernel against a scalar CPU
/// reference built from `QuantizationSubByte.dequantizeInt5Affine`. Proves
/// the kernel's sub-byte unpacking matches the CPU reference's bit-packing
/// exactly (any mismatch in the straddling-byte bit math would show up as a
/// large relative error here, not just a rounding-level one).
@Suite struct DequantInt5GEMVTests {

    private static func scalarRef(
        weightRows: [QuantizationSubByte.Int5AffineRow], x: [Float], n: Int
    ) -> [Float] {
        weightRows.map { row in
            let w = QuantizationSubByte.dequantizeInt5Affine(row, n: n)
            var sum: Float = 0
            for i in 0..<n { sum += w[i] * x[i] }
            return sum
        }
    }

    private static func runAndCompare(m: Int, n: Int, groupSize: Int, seed: UInt64) throws -> Float {
        precondition(n % groupSize == 0)
        let groupsPerRow = n / groupSize
        var rng = SeedTree(seed).key("int5-gemv-m\(m)-n\(n)-g\(groupSize)")

        var rows: [QuantizationSubByte.Int5AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            rows.append(QuantizationSubByte.quantizeInt5Affine(raw, groupSize: groupSize))
        }

        let rowBytes = (n * 5) / 8
        var packed = [UInt8](repeating: 0, count: m * rowBytes)
        var scales = [UInt16](repeating: 0, count: m * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: m * groupsPerRow)
        for row in 0..<m {
            for i in 0..<rowBytes { packed[row * rowBytes + i] = rows[row].packed[i] }
            for g in 0..<groupsPerRow {
                scales[row * groupsPerRow + g] = rows[row].scales[g]
                biases[row * groupsPerRow + g] = rows[row].biases[g]
            }
        }

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantInt5GEMV(context: ctx)

        guard let wBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return .infinity
        }
        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return .infinity
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, scales: sBuf, biases: bBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n), groupSize: UInt32(groupSize))
        cmd.commit()
        cmd.waitUntilCompleted()

        let ref = Self.scalarRef(weightRows: rows, x: xRef, n: n)
        let actual = Fp16Buffer.read(yBuf, count: m)
        return RelError.compute(actual: actual, reference: ref)
    }

    @Test func gemv_layer0ProjectionShape() throws {
        // Representative layer-0 projection shape, group_size=64.
        let rel = try Self.runAndCompare(m: 128, n: 2816, groupSize: 64, seed: 0x551)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }

    @Test func gemv_groupSize128() throws {
        let rel = try Self.runAndCompare(m: 64, n: 1024, groupSize: 128, seed: 0x552)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }

    @Test(arguments: [8, 32, 64] as [Int], [64, 128, 320] as [Int])
    func gemv_sweep(m: Int, n: Int) throws {
        let rel = try Self.runAndCompare(m: m, n: n, groupSize: 64, seed: UInt64(m * 1000 + n))
        #expect(rel < Tolerance.fp16Reduction, "M=\(m) N=\(n) rel=\(rel)")
    }
}
