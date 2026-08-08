import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P2-6: Layer-3 weight validation against the repacked Qwen3.6 model.
///
/// Full forward-pass comparison against the P0-7 fixture requires Multi-GPU
/// Metal orchestration. These tests verify layer 3's weights load correctly.
@Suite struct Qwen36Layer3IsolationTests {
    private static let modelDir = "/tmp/qwen36.gturbo"

    private static var isModelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir)
    }

    private static func loadModel() throws -> Model {
        let device = MTLCreateSystemDefaultDevice()!
        return try Model.load(directoryURL: URL(fileURLWithPath: modelDir),
                              device: device, expecting: .qwen36_35B_A3B)
    }

    @Test func configIsQwen36() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        #expect(m.config.topology == .qwen36)
        #expect(m.config.activation == .silu)
        #expect(m.config.numLayers == 40)
        #expect(m.config.hiddenSize == 2048)
    }

    @Test func layer3IsFullAttention() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        #expect(m.config.layerKindMask[3] == 1)
    }

    @Test func layer3RouterWeightShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.router(layer: 3)
        #expect(w.shape.0 == 256)
        #expect(w.shape.1 == 2048)
    }

    @Test func layer3QProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let q = try m.qProj(layer: 3)
        #expect(q.shape.0 == 8192)
        #expect(q.shape.1 == 2048)
    }

    @Test func layer3KProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let k = try m.kProj(layer: 3)
        #expect(k.shape.0 == 512)
        #expect(k.shape.1 == 2048)
    }

    @Test func layer3OProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let o = try m.oProj(layer: 3)
        #expect(o.shape.0 == 2048)   // hiddenSize
        #expect(o.shape.1 == 4096)   // numHeads×headDim = 16×256
    }

    @Test func layer3SharedExpertGateShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let g = try m.sharedExpertGate(layer: 3)
        #expect(g.shape.0 == 512)
        #expect(g.shape.1 == 2048)
    }

    @Test func layer3NormsLoad() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        _ = try m.inputNorm(layer: 3)
        _ = try m.postAttnNorm(layer: 3)
        _ = try m.qNorm(layer: 3)
        _ = try m.kNorm(layer: 3)
    }
}
