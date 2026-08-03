import Testing
import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

@Suite struct DenseMLPLayerTests {

    @Test func configIdentifiesDenseMLPLayerCorrectly() {
        let denseMask: [UInt8] = [1, 0, 0, 0]
        let cfg = ArchConfig(
            hiddenSize: 64,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 128,
            slidingWindow: 16,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1, 0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            denseMLPLayerMask: denseMask,
            denseMLPIntermediateSize: 128
        )

        #expect(cfg.isDenseMLP(layer: 0) == true)
        #expect(cfg.isDenseMLP(atLayer: 0) == true)
        #expect(cfg.isDenseMLP(layer: 1) == false)
        #expect(cfg.isDenseMLP(atLayer: 1) == false)
        #expect(cfg.isDenseMLP(layer: 2) == false)
        #expect(cfg.isDenseMLP(layer: 3) == false)
        #expect(cfg.numSparseLayers == 3)
    }

    @Test func prefillChunkScratchLayoutSizesSharedIntermediateToMax() {
        let cfg = ArchConfig(
            hiddenSize: 64,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 128,
            slidingWindow: 16,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            denseMLPLayerMask: [1, 0],
            denseMLPIntermediateSize: 128
        )

        let layout = PrefillChunkScratchLayout(config: cfg, chunkTokens: 16)
        #expect(layout.sharedIntermediate == 128)
        #expect(layout.sharedExpertScratchElements == 128)
    }

    /// Build a minimal synthetic model directory where Layer 0 is dense MLP (no router)
    /// and Layer 1 is MoE (with router), verifying RealForwardRunner init, prefill, and decode.
    static func writeDenseToySynthetic() throws -> (URL, ArchConfig) {
        let denseMask: [UInt8] = [1, 0]
        let toy = ArchConfig(
            hiddenSize: 64,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 128,
            slidingWindow: 16,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            denseMLPLayerMask: denseMask,
            denseMLPIntermediateSize: 128
        )

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-dense-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let embedSize = UInt64(toy.vocabSize * toy.hiddenSize)
        let bf16DBytes = UInt64(d * MemoryLayout<UInt16>.stride)

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = (cols + Quantization.groupSize - 1) / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001
                        + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func appendProjection(rows: [[Float]], to bytes: inout [UInt8], component: String) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            switch component {
            case "packed":
                for row in quantized { bytes.append(contentsOf: row.packed) }
            case "scales":
                for row in quantized { appendU16(row.scales, to: &bytes) }
            case "biases":
                for row in quantized { appendU16(row.biases, to: &bytes) }
            default:
                preconditionFailure("unknown projection component \(component)")
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols, expert: expert, role: role)
                let packedOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "packed")
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": 4,
                ]
                let scalesOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "scales")
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "biases")
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d, role: 1)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        var specs: [ResidentSpec] = [
            ResidentSpec(name: "language_model.model.embed_tokens.weight",
                         dtype: 0,
                         shape: [UInt32(toy.vocabSize), UInt32(toy.hiddenSize), 0, 0],
                         weightBytes: embedSize,
                         scaleBytes: UInt64(toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride),
                         biasBytes: UInt64(toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride)),
            ResidentSpec(name: "language_model.model.norm.weight",
                         dtype: 1,
                         shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                         weightBytes: bf16DBytes,
                         scaleBytes: 0,
                         biasBytes: 0),
        ]
        for L in 0..<toy.numLayers {
            let isFull = toy.fullAttentionLayerMask[L] != 0
            let isDense = toy.isDenseMLP(atLayer: L)
            let headDim = isFull ? toy.fullHeadDim : toy.headDim
            let numKVHeads = isFull ? toy.numFullKVHeads : toy.numKVHeads
            let qDim = toy.numHeads * headDim
            let kvDim = numKVHeads * headDim
            let layerF = isDense ? toy.denseMLPIntermediateSize : toy.intermediateSize

            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).input_layernorm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.q_proj.weight", rows: qDim, cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.k_proj.weight", rows: kvDim, cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.v_proj.weight", rows: kvDim, cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.o_proj.weight", rows: d, cols: qDim))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_attention_layernorm.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).self_attn.q_norm.weight",
                dtype: 1, shape: [UInt32(headDim), 0, 0, 0], weightBytes: UInt64(headDim * MemoryLayout<UInt16>.stride), scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).self_attn.k_norm.weight",
                dtype: 1, shape: [UInt32(headDim), 0, 0, 0], weightBytes: UInt64(headDim * MemoryLayout<UInt16>.stride), scaleBytes: 0, biasBytes: 0))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.gate_proj.weight", rows: layerF, cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.up_proj.weight", rows: layerF, cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.down_proj.weight", rows: d, cols: layerF))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight",
                dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).layer_scalar",
                dtype: 1, shape: [1, 0, 0, 0], weightBytes: UInt64(MemoryLayout<UInt16>.stride), scaleBytes: 0, biasBytes: 0))

            if !isDense {
                specs.append(ResidentSpec(
                    name: "language_model.model.layers.\(L).router.scale",
                    dtype: 1, shape: [UInt32(toy.hiddenSize), 0, 0, 0], weightBytes: bf16DBytes, scaleBytes: 0, biasBytes: 0))
                specs.append(int4AffineSpec(
                    "language_model.model.layers.\(L).router.proj.weight", rows: toy.numExperts, cols: d))
                specs.append(ResidentSpec(
                    name: "language_model.model.layers.\(L).router.per_expert_scale",
                    dtype: 1, shape: [UInt32(toy.numExperts), 0, 0, 0], weightBytes: UInt64(toy.numExperts * MemoryLayout<UInt16>.stride), scaleBytes: 0, biasBytes: 0))
            }
        }

        var resData = [UInt8]()
        var indexEntries: [String: Any] = [:]

        func writeHeader() {
            let magic: UInt32 = 0x42525554
            let version: UInt32 = 1
            appendU32(magic, to: &resData)
            appendU32(version, to: &resData)
            appendU32(0, to: &resData) // placeholder for indexSize
        }
        func appendU32(_ v: UInt32, to bytes: inout [UInt8]) {
            bytes.append(contentsOf: withUnsafeBytes(of: v.littleEndian) { Array($0) })
        }

        writeHeader()

        for spec in specs {
            let tensorStart = UInt64(resData.count)
            resData.append(contentsOf: Array(repeating: UInt8(0x01), count: Int(spec.weightBytes)))
            var auxStart: UInt64 = 0
            if spec.scaleBytes > 0 {
                auxStart = UInt64(resData.count)
                resData.append(contentsOf: Array(repeating: UInt8(0x3C), count: Int(spec.scaleBytes)))
                resData.append(contentsOf: Array(repeating: UInt8(0x00), count: Int(spec.biasBytes)))
            }
            indexEntries[spec.name] = [
                "offset": tensorStart,
                "size": spec.weightBytes,
                "dtype": spec.dtype == 0 ? "U32" : "BF16",
                "shape": spec.shape.filter { $0 != 0 },
                "bits": spec.dtype == 0 ? 4 : 16,
                "scaleOffset": auxStart,
                "biasOffset": auxStart + spec.scaleBytes,
            ]
        }

        let archiveDict: [String: Any] = [
            "format": "gturbo",
            "version": 1,
            "architecture": [
                "numLayers": toy.numLayers,
                "hiddenSize": toy.hiddenSize,
                "intermediateSize": toy.intermediateSize,
                "moeIntermediateSize": toy.moeIntermediateSize,
                "numHeads": toy.numHeads,
                "numKVHeads": toy.numKVHeads,
                "numFullKVHeads": toy.numFullKVHeads,
                "headDim": toy.headDim,
                "fullHeadDim": toy.fullHeadDim,
                "vocabSize": toy.vocabSize,
                "slidingWindow": toy.slidingWindow,
                "finalLogitSoftcap": toy.finalLogitSoftcap,
                "ropeTheta": toy.ropeTheta,
                "fullRopeTheta": toy.fullRopeTheta,
                "partialRotaryFactor": toy.partialRotaryFactor,
                "numExperts": toy.numExperts,
                "topKExperts": toy.topKExperts,
                "tieWordEmbeddings": toy.tieWordEmbeddings,
                "attentionKEqV": toy.attentionKEqV,
                "fullAttentionLayerMask": toy.fullAttentionLayerMask,
                "denseMLPLayerMask": toy.denseMLPLayerMask,
                "denseMLPIntermediateSize": toy.denseMLPIntermediateSize,
                "hiddenActivation": toy.hiddenActivation,
            ] as [String: Any],
            "quantization": [
                "embedding": ["weightBits": 4, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
                "attention": ["weightBits": 4, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
                "sharedExpert": ["weightBits": 4, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
                "routedExperts": ["weightBits": 4, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
                "router": ["weightBits": 4, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
            ] as [String: Any],
            "index": indexEntries,
        ]

        let jsonBytes = try JSONSerialization.data(withJSONObject: archiveDict, options: [.prettyPrinted, .sortedKeys])
        let indexLen = UInt32(jsonBytes.count)
        resData.withUnsafeMutableBytes { ptr in
            let raw = ptr.baseAddress!.assumingMemoryBound(to: UInt32.self)
            raw[2] = indexLen.littleEndian
        }
        resData.append(contentsOf: jsonBytes)

        let resURL = dir.appendingPathComponent("resident.bin")
        try Data(resData).write(to: resURL)

        // Packed experts layout & data for layer 1 (MoE)
        var layoutLayers: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            let filename = String(format: "layer_%02d.bin", L)
            if toy.isDenseMLP(atLayer: L) {
                layoutLayers.append([
                    "layer": L,
                    "file": "",
                    "offset": 0,
                    "size": 0,
                    "experts": [],
                ])
                continue
            }
            let blobPath = exp.appendingPathComponent(filename)
            var blobData = [UInt8]()
            var expOffsets: [[String: Any]] = []

            for e in 0..<toy.numExperts {
                let (ebytes, etensors) = toyExpertBlob(expert: e)
                let start = blobData.count
                blobData.append(contentsOf: ebytes)
                expOffsets.append([
                    "expert": e,
                    "offset": start,
                    "size": ebytes.count,
                    "tensors": etensors,
                ])
            }
            try Data(blobData).write(to: blobPath)
            layoutLayers.append([
                "layer": L,
                "file": filename,
                "offset": 0,
                "size": blobData.count,
                "experts": expOffsets,
            ])
        }

        let layoutDict: [String: Any] = [
            "expertStride": 16384,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layoutLayers,
        ]
        let layoutData = try JSONSerialization.data(withJSONObject: layoutDict, options: [.prettyPrinted, .sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)

        // Write manifest.json
        var files: [String: [String: Any]] = [
            "resident.bin": ["size": resData.count, "sha256": try Sha256Verifier.hashFile(at: resURL)],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": try Sha256Verifier.hashFile(at: layoutURL)],
        ]
        for L in 0..<toy.numLayers {
            if !toy.isDenseMLP(atLayer: L) {
                let name = String(format: "layer_%02d.bin", L)
                let url = exp.appendingPathComponent(name)
                files["packed_experts/\(name)"] = ["size": try Data(contentsOf: url).count, "sha256": try Sha256Verifier.hashFile(at: url)]
            }
        }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "denseMLPLayerMask": toy.denseMLPLayerMask.map { Int($0) },
            "denseMLPIntermediateSize": toy.denseMLPIntermediateSize,
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot, options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))

        try ModelLoaderTests.writeVerifiedInstallReceipt(directoryURL: dir)

        return (dir, toy)
    }

    @Test func denseLayer0ForwardPassInPrefillAndDecode() async throws {
        let (dir, toy) = try Self.writeDenseToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let model = try Model.load(directoryURL: dir, device: device, expecting: toy)
        let ctx = try MetalContext()
        let runner = try RealForwardRunner(model: model, context: ctx, maxContext: 128)

        // 1. Test Prefill forward pass through dense layer 0 & routed layer 1
        let tokens: [Int32] = [1, 5, 12, 42]
        guard let logits = ctx.device.makeBuffer(length: model.config.vocabSize * MemoryLayout<Float16>.stride,
                                                 options: .storageModeShared) else {
            return
        }

        let result = try await runner.prefillChunked(tokens: tokens[...],
                                                    startPosition: 0,
                                                    outputMode: .logits,
                                                    config: .defaultChunked,
                                                    into: logits,
                                                    onProgress: { _ in })
        #expect(result.newPosition == 4)

        let ptr = logits.contents().assumingMemoryBound(to: Float16.self)
        let val0 = Float(ptr[0])
        #expect(!val0.isNaN, "Prefill logit at index 0 should not be NaN")
        #expect(!val0.isInfinite, "Prefill logit at index 0 should not be Inf")

        // 2. Test Decode forward pass through dense layer 0 & routed layer 1
        try await runner.produce(token: 42, position: 4, into: logits)
        let valDec = Float(ptr[0])
        #expect(!valDec.isNaN, "Decode logit at index 0 should not be NaN")
        #expect(!valDec.isInfinite, "Decode logit at index 0 should not be Inf")
    }
}
