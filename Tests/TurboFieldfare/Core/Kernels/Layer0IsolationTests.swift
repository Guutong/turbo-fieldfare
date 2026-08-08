import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P3-1: Layer-0 weight validation — the first layer is DeltaNet, not full attention.
///
/// These tests verify that layer 0's weights load correctly from the repacked
/// Qwen3.6-35B-A3B checkpoint and have the expected shapes.
@Suite struct Qwen36Layer0IsolationTests {
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

    @Test func layer0IsDeltaNet() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        #expect(m.config.layerKindMask[0] == 2)
    }

    @Test func layer0QKVProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaQKVProj(layer: 0)
        #expect(w.shape.0 == 8192)
        #expect(w.shape.1 == 2048)
    }

    @Test func layer0ZProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaZProj(layer: 0)
        #expect(w.shape.0 == 4096)
        #expect(w.shape.1 == 2048)
    }

    @Test func layer0AProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaAProj(layer: 0)
        #expect(w.shape.0 == 32)
        #expect(w.shape.1 == 2048)
    }

    @Test func layer0BProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaBProj(layer: 0)
        #expect(w.shape.0 == 32)
        #expect(w.shape.1 == 2048)
    }

    @Test func layer0OutProjShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaOutProj(layer: 0)
        #expect(w.shape.0 == 2048)   // hiddenSize
        #expect(w.shape.1 == 4096)   // numHeads × headDim
    }

    @Test func layer0Conv1dShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaConv1d(layer: 0)
        #expect(w.shape.0 == 8192)
        #expect(w.shape.1 == 4)
    }

    @Test func layer0NormShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaNorm(layer: 0)
        #expect(w.shape.0 == 128)
    }

    @Test func layer0ALogShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaALog(layer: 0)
        #expect(w.shape.0 == 32)
    }

    @Test func layer0DtBiasShape() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaDtBias(layer: 0)
        #expect(w.shape.0 == 32)
    }

    @Test func layer0AffineHasScalesAndBiases() throws {
        guard Self.isModelAvailable else { return }
        let m = try Self.loadModel()
        let w = try m.deltaQKVProj(layer: 0)
        // INT4 quantisation stores scales and biases alongside the compressed weights
        #expect(w.scaleLength > 0)
        #expect(w.biasLength > 0)
    }
}
