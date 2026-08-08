import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P3-3: Layer-0 isolation parity test — the DeltaNet analogue of P2-6's
/// injection trick.
///
/// Loads the P0-7 fixture, injects the recorded `hidden_in.0` at the input of
/// layer 0 (a DeltaNet layer), runs ONLY layer 0's forward pass (input norm ->
/// DeltaNet block -> residual -> post-attn norm -> shared+routed MoE ->
/// residual) in plain Swift fp32 against the real repacked weights, and
/// compares against the fixture's recorded `hidden_out.0` and `expert_ids.0`.
///
/// This is deliberately NOT wired through `RealForwardRunner` / Metal: Phase 3
/// is plain-Swift-fp32-first (ADR-0001), and no Metal orchestration for
/// per-layer isolated forward passes exists yet (P2-6 stayed shape-only for
/// the same reason). The int4/int8 dequantization here mirrors the on-disk
/// affine-quantization scheme documented in `Quantization.swift` and cross-
/// checked directly against the resident index / packed-experts layout.
@Suite struct Qwen36Layer0NumericIsolationTests {
    private static let modelDir = "scratch/qwen36.gturbo"
    private static let fixturePath = "Tests/Fixtures/qwen36_fixture.safetensors"

    private static var isModelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir)
    }
    private static var isFixtureAvailable: Bool {
        FileManager.default.fileExists(atPath: fixturePath)
    }

    private static func loadModel() throws -> Model {
        let device = MTLCreateSystemDefaultDevice()!
        return try Model.load(directoryURL: URL(fileURLWithPath: modelDir),
                              device: device, expecting: .qwen36_35B_A3B)
    }

    // MARK: - Minimal safetensors reader (fixture only; self-contained)

    private struct FixtureEntry {
        let dtype: String
        let shape: [Int]
        let start: Int
        let end: Int
    }

    private struct Fixture {
        let raw: Data
        let payloadBase: Int
        let entries: [String: FixtureEntry]

        init(path: String) throws {
            let raw = try Data(contentsOf: URL(fileURLWithPath: path))
            let headerLen = raw.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            let hdrStart = 8
            let hdrEnd = 8 + Int(headerLen)
            let headerData = raw.subdata(in: hdrStart..<hdrEnd)
            let obj = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
            var entries: [String: FixtureEntry] = [:]
            for (k, v) in obj {
                if k == "__metadata__" { continue }
                guard let d = v as? [String: Any],
                      let dtype = d["dtype"] as? String,
                      let shapeAny = d["shape"] as? [Any],
                      let offs = d["data_offsets"] as? [Any], offs.count == 2 else { continue }
                let shape = shapeAny.map { ($0 as! NSNumber).intValue }
                let start = (offs[0] as! NSNumber).intValue
                let end = (offs[1] as! NSNumber).intValue
                entries[k] = FixtureEntry(dtype: dtype, shape: shape, start: start, end: end)
            }
            self.raw = raw
            self.payloadBase = hdrEnd
            self.entries = entries
        }

        func floats(_ name: String) -> [Float] {
            guard let e = entries[name] else {
                fatalError("fixture missing tensor \(name)")
            }
            precondition(e.dtype == "F32", "\(name) is \(e.dtype), expected F32")
            let slice = raw.subdata(in: (payloadBase + e.start)..<(payloadBase + e.end))
            return slice.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
        }

        func u32s(_ name: String) -> [UInt32] {
            guard let e = entries[name] else {
                fatalError("fixture missing tensor \(name)")
            }
            precondition(e.dtype == "U32", "\(name) is \(e.dtype), expected U32")
            let slice = raw.subdata(in: (payloadBase + e.start)..<(payloadBase + e.end))
            return slice.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: UInt32.self))
            }
        }
    }

    // MARK: - CPU dequant helpers (affine int4/int8, groupSize 64, row-major)

    /// Dequantize a resident affine-quantized matrix (weights/scales/biases in
    /// the SAME buffer, offsets from a `TensorView`) to row-major `[Float]`.
    private static func dequantResident(_ tv: TensorView, rows: Int, cols: Int, bits: Int) -> [Float] {
        dequantAffine(buffer: tv.buffer,
                      weightOffset: Int(tv.offset),
                      scaleOffset: Int(tv.scaleOffset),
                      biasOffset: Int(tv.biasOffset),
                      rows: rows, cols: cols, bits: bits)
    }

    private static func dequantAffine(buffer: MTLBuffer, weightOffset: Int,
                                      scaleOffset: Int, biasOffset: Int,
                                      rows: Int, cols: Int, bits: Int) -> [Float] {
        let groupSize = 64
        let groupsPerRow = cols / groupSize
        let base = buffer.contents()
        let scalePtr = base.advanced(by: scaleOffset).assumingMemoryBound(to: UInt16.self)
        let biasPtr = base.advanced(by: biasOffset).assumingMemoryBound(to: UInt16.self)
        var out = [Float](repeating: 0, count: rows * cols)
        if bits == 4 {
            let wPtr = base.advanced(by: weightOffset).assumingMemoryBound(to: UInt8.self)
            let bytesPerRow = cols / 2
            for r in 0..<rows {
                for g in 0..<groupsPerRow {
                    let scale = Quantization.bf16ToFloat(scalePtr[r * groupsPerRow + g])
                    let bias = Quantization.bf16ToFloat(biasPtr[r * groupsPerRow + g])
                    for k in 0..<groupSize {
                        let c = g * groupSize + k
                        let byte = wPtr[r * bytesPerRow + c / 2]
                        let nibble = (c & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                        out[r * cols + c] = Float(nibble) * scale + bias
                    }
                }
            }
        } else {
            precondition(bits == 8)
            let wPtr = base.advanced(by: weightOffset).assumingMemoryBound(to: UInt8.self)
            for r in 0..<rows {
                for g in 0..<groupsPerRow {
                    let scale = Quantization.bf16ToFloat(scalePtr[r * groupsPerRow + g])
                    let bias = Quantization.bf16ToFloat(biasPtr[r * groupsPerRow + g])
                    for k in 0..<groupSize {
                        let c = g * groupSize + k
                        out[r * cols + c] = Float(wPtr[r * cols + c]) * scale + bias
                    }
                }
            }
        }
        return out
    }

    private static func bf16Vector(_ tv: TensorView, count: Int) -> [Float] {
        let ptr = tv.buffer.contents().advanced(by: Int(tv.offset)).assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Quantization.bf16ToFloat(ptr[$0]) }
    }

    // MARK: - Plain math helpers

    private static func matVec(_ w: [Float], rows: Int, cols: Int, _ x: [Float]) -> [Float] {
        precondition(w.count == rows * cols)
        precondition(x.count == cols)
        var y = [Float](repeating: 0, count: rows)
        w.withUnsafeBufferPointer { wp in
            x.withUnsafeBufferPointer { xp in
                for r in 0..<rows {
                    var acc: Float = 0
                    let base = r * cols
                    for c in 0..<cols { acc += wp[base + c] * xp[c] }
                    y[r] = acc
                }
            }
        }
        return y
    }

    private static func rmsNorm(_ x: [Float], weight: [Float], eps: Float = 1e-6) -> [Float] {
        let n = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / sqrt(ss / Float(n) + eps)
        return (0..<n).map { x[$0] * inv * weight[$0] }
    }

    private static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }
    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    private static func softmax(_ x: [Float]) -> [Float] {
        let m = x.max()!
        let exps = x.map { exp($0 - m) }
        let s = exps.reduce(0, +)
        return exps.map { $0 / s }
    }

    // MARK: - The test

    @Test func layer0ForwardMatchesFixtureWithinTolerance() throws {
        guard Self.isModelAvailable, Self.isFixtureAvailable else { return }

        let model = try Self.loadModel()
        let fixture = try Fixture(path: Self.fixturePath)

        let D = 2048
        let numTokens = 5
        let hiddenIn = fixture.floats("hidden_in.0")     // [5, 2048]
        let hiddenOutExpected = fixture.floats("hidden_out.0")
        let expertIdsExpected = fixture.u32s("expert_ids.0") // [5, 8]
        #expect(hiddenIn.count == numTokens * D)
        #expect(hiddenOutExpected.count == numTokens * D)

        // --- Weights (dequantized once, reused across all 5 tokens) ---
        let inputNormW = Self.bf16Vector(try model.inputNorm(layer: 0), count: D)
        let postAttnNormW = Self.bf16Vector(try model.postAttnNorm(layer: 0), count: D)

        let convDim = 8192
        let keyDim = 2048
        let valueDim = 4096
        let numKeyHeads = 16
        let numValueHeads = 32
        let headKDim = 128
        let headVDim = 128

        let qkvW = Self.dequantResident(try model.deltaQKVProj(layer: 0), rows: convDim, cols: D, bits: 4)
        let zW = Self.dequantResident(try model.deltaZProj(layer: 0), rows: valueDim, cols: D, bits: 4)
        let aW = Self.dequantResident(try model.deltaAProj(layer: 0), rows: numValueHeads, cols: D, bits: 4)
        let bW = Self.dequantResident(try model.deltaBProj(layer: 0), rows: numValueHeads, cols: D, bits: 4)
        let outW = Self.dequantResident(try model.deltaOutProj(layer: 0), rows: D, cols: valueDim, bits: 4)
        let convW = Self.bf16Vector(try model.deltaConv1d(layer: 0), count: convDim * 4)
        let deltaNormW = Self.bf16Vector(try model.deltaNorm(layer: 0), count: headVDim)
        let aLog = Self.bf16Vector(try model.deltaALog(layer: 0), count: numValueHeads)
        let dtBias = Self.bf16Vector(try model.deltaDtBias(layer: 0), count: numValueHeads)

        let numExperts = 256
        let moeIntermediate = 512
        let topK = 8
        let routerW = Self.dequantResident(try model.router(layer: 0), rows: numExperts, cols: D, bits: 8)
        let sharedGateW = Self.dequantResident(try model.sharedExpertGate(layer: 0), rows: moeIntermediate, cols: D, bits: 4)
        let sharedUpW = Self.dequantResident(try model.sharedExpertUp(layer: 0), rows: moeIntermediate, cols: D, bits: 4)
        let sharedDownW = Self.dequantResident(try model.sharedExpertDown(layer: 0), rows: D, cols: moeIntermediate, bits: 4)
        // `mlp.shared_expert_gate` (scalar sigmoid gate) — no dedicated Model
        // accessor exists yet; read the resident tensor directly (int8-affine,
        // same as the router, per manifest.quant).
        let sharedExpertGateScalarW = Self.dequantResident(
            try model.resident(name: "language_model.model.layers.0.mlp.shared_expert_gate.weight"),
            rows: 1, cols: D, bits: 8)

        // --- Sequential state (fresh for this isolated 5-token run) ---
        var convState = [Float](repeating: 0, count: 3 * convDim)
        var recurrentState = [Float](repeating: 0, count: numValueHeads * headVDim * headKDim)

        var hiddenOutActual = [Float](repeating: 0, count: numTokens * D)
        var expertIdsActual: [[UInt32]] = []

        for t in 0..<numTokens {
            let x = Array(hiddenIn[(t * D)..<((t + 1) * D)])

            // --- DeltaNet block ---
            let a = Self.rmsNorm(x, weight: inputNormW)
            let qkv = Self.matVec(qkvW, rows: convDim, cols: D, a)
            let z = Self.matVec(zW, rows: valueDim, cols: D, a)
            let aGateRaw = Self.matVec(aW, rows: numValueHeads, cols: D, a)
            let bGateRaw = Self.matVec(bW, rows: numValueHeads, cols: D, a)

            let convOut = DeltaNetConv.step(input: qkv, weight: convW, convDim: convDim, convState: &convState)

            var q = Array(convOut[0..<keyDim])
            var k = Array(convOut[keyDim..<(2 * keyDim)])
            let v = Array(convOut[(2 * keyDim)..<(2 * keyDim + valueDim)])

            DeltaNetQKNorm.applyInPlace(&q, numHeads: numKeyHeads, headDim: headKDim,
                                        scale: DeltaNetQKNorm.qScale(headKDim: headKDim))
            DeltaNetQKNorm.applyInPlace(&k, numHeads: numKeyHeads, headDim: headKDim,
                                        scale: DeltaNetQKNorm.kScale(headKDim: headKDim))

            let qExp = DeltaNetHeadExpansion.expand(q, numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, headDim: headKDim)
            let kExp = DeltaNetHeadExpansion.expand(k, numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, headDim: headKDim)

            let beta = DeltaNetGate.beta(bGateRaw)
            let g = DeltaNetGate.decay(a: aGateRaw, aLog: aLog, dtBias: dtBias)

            let y = DeltaNetRecurrence.step(q: qExp, k: kExp, v: v, beta: beta, g: g,
                                            numValueHeads: numValueHeads, headVDim: headVDim, headKDim: headKDim,
                                            state: &recurrentState)

            let gated = DeltaNetOutputGate.apply(y: y, z: z, normWeight: deltaNormW,
                                                 numValueHeads: numValueHeads, headVDim: headVDim)

            let deltaOut = Self.matVec(outW, rows: D, cols: valueDim, gated)

            let h = zip(x, deltaOut).map(+)

            // --- MLP block (router + shared expert + routed MoE) ---
            let hp = Self.rmsNorm(h, weight: postAttnNormW)

            let routerLogits = Self.matVec(routerW, rows: numExperts, cols: D, hp)
            let probs = Self.softmax(routerLogits)
            if ProcessInfo.processInfo.environment["L0_DEBUG"] != nil {
                let fixtureProbs = fixture.floats("router_logits.0")
                var pSSD: Double = 0, pSSE: Double = 0
                for i in 0..<numExperts {
                    let expected = fixtureProbs[t * numExperts + i]
                    let diff = probs[i] - expected
                    pSSD += Double(diff * diff)
                    pSSE += Double(expected * expected)
                }
                print("DEBUG token \(t): router-probs relL2 = \(sqrt(pSSD / max(pSSE, 1e-12)))")
            }
            let sortedIdx = probs.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
            let top = Array(sortedIdx.prefix(topK))
            let topSum = top.reduce(Float(0)) { $0 + probs[$1] }
            let topScores = top.map { probs[$0] / topSum }

            let sg = Self.matVec(sharedGateW, rows: moeIntermediate, cols: D, hp)
            let su = Self.matVec(sharedUpW, rows: moeIntermediate, cols: D, hp)
            let sharedAct = zip(sg, su).map { Self.silu($0) * $1 }
            var sharedY = Self.matVec(sharedDownW, rows: D, cols: moeIntermediate, sharedAct)
            let gateScalarLogit = Self.matVec(sharedExpertGateScalarW, rows: 1, cols: D, hp)[0]
            let sharedGateVal = Self.sigmoid(gateScalarLogit)
            sharedY = sharedY.map { $0 * sharedGateVal }

            var routedY = [Float](repeating: 0, count: D)
            for (rank, expertIdx) in top.enumerated() {
                let weight = topScores[rank]
                let expertTV = try model.routedExpert(layer: 0, expert: expertIdx)
                let entry = model.packedExpertsLayout.expert(layer: 0, expert: expertIdx)
                let base = Int(expertTV.offset)

                func sub(_ role: String) -> (weightOffset: Int, scaleOffset: Int, biasOffset: Int, rows: Int, cols: Int) {
                    let w = entry.subTensors[role]!
                    let s = entry.subTensors["\(role)_scales"]!
                    let b = entry.subTensors["\(role)_biases"]!
                    return (base + Int(w.offset), base + Int(s.offset), base + Int(b.offset),
                           Int(w.shape[0]), Int(w.shape[1]))
                }
                let gateInfo = sub("gate")
                let upInfo = sub("up")
                let downInfo = sub("down")

                let gateW = Self.dequantAffine(buffer: expertTV.buffer, weightOffset: gateInfo.weightOffset,
                                               scaleOffset: gateInfo.scaleOffset, biasOffset: gateInfo.biasOffset,
                                               rows: gateInfo.rows, cols: gateInfo.cols, bits: 4)
                let upW2 = Self.dequantAffine(buffer: expertTV.buffer, weightOffset: upInfo.weightOffset,
                                              scaleOffset: upInfo.scaleOffset, biasOffset: upInfo.biasOffset,
                                              rows: upInfo.rows, cols: upInfo.cols, bits: 4)
                let downW2 = Self.dequantAffine(buffer: expertTV.buffer, weightOffset: downInfo.weightOffset,
                                                scaleOffset: downInfo.scaleOffset, biasOffset: downInfo.biasOffset,
                                                rows: downInfo.rows, cols: downInfo.cols, bits: 4)

                let eg = Self.matVec(gateW, rows: gateInfo.rows, cols: gateInfo.cols, hp)
                let eu = Self.matVec(upW2, rows: upInfo.rows, cols: upInfo.cols, hp)
                let eAct = zip(eg, eu).map { Self.silu($0) * $1 }
                let eOut = Self.matVec(downW2, rows: downInfo.rows, cols: downInfo.cols, eAct)
                for i in 0..<D { routedY[i] += weight * eOut[i] }
            }

            let mlpOut = zip(routedY, sharedY).map(+)
            let hOut = zip(h, mlpOut).map(+)

            for i in 0..<D { hiddenOutActual[t * D + i] = hOut[i] }
            expertIdsActual.append(top.map { UInt32($0) })
        }

        // --- Compare against the fixture ---
        var sumSqDiff: Double = 0
        var sumSqExpected: Double = 0
        var maxAbs: Float = 0
        for t in 0..<numTokens {
            var tSSD: Double = 0
            var tSSE: Double = 0
            var tMax: Float = 0
            for i in 0..<D {
                let idx = t * D + i
                let diff = hiddenOutActual[idx] - hiddenOutExpected[idx]
                tSSD += Double(diff * diff)
                tSSE += Double(hiddenOutExpected[idx] * hiddenOutExpected[idx])
                tMax = max(tMax, abs(diff))
            }
            let tRel = sqrt(tSSD / max(tSSE, 1e-12))
            print("DEBUG token \(t): relL2=\(tRel) maxAbs=\(tMax)")
        }
        for i in 0..<(numTokens * D) {
            let diff = hiddenOutActual[i] - hiddenOutExpected[i]
            sumSqDiff += Double(diff * diff)
            sumSqExpected += Double(hiddenOutExpected[i] * hiddenOutExpected[i])
            maxAbs = max(maxAbs, abs(diff))
        }
        let relL2 = Float(sqrt(sumSqDiff / max(sumSqExpected, 1e-12)))

        for t in 0..<numTokens {
            let expected = Set(expertIdsExpected[(t * 8)..<((t + 1) * 8)])
            let actual = Set(expertIdsActual[t])
            #expect(actual == expected,
                   "token \(t): expert set mismatch — expected \(expected.sorted()), got \(actual.sorted())")
        }

        #expect(relL2 <= 2e-2, "rel-L2 \(relL2) exceeds gate 2e-2 (ADR-0002 — do not widen)")
        #expect(maxAbs <= 1e-2, "max-abs \(maxAbs) exceeds gate 1e-2 (ADR-0002 — do not widen)")
    }
}
