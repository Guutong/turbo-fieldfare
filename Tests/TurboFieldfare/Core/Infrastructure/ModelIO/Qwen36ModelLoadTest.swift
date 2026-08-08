import Foundation
import Metal
import Testing
@testable import TurboFieldfare

struct Qwen36ModelLoadTest {
    @Test func loadRepackedModel() throws {
        let modelDir = URL(fileURLWithPath: "/tmp/qwen36.gturbo")
        let device = MTLCreateSystemDefaultDevice()!
        let model = try Model.load(directoryURL: modelDir,
                                   device: device,
                                   expecting: .qwen36_35B_A3B)
        #expect(model.config.topology == .qwen36)
        #expect(model.config.activation == .silu)
        #expect(model.config.numLayers == 40)
        #expect(model.config.numExperts == 256)
        #expect(model.config.hiddenSize == 2048)
    }
}
