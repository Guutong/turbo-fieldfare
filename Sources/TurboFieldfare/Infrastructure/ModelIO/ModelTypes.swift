import Foundation
import Metal

/// YaRN RoPE scaling parameters. Applied per layer type: Laguna-S-2.1 scales
/// its full-attention layers with YaRN while leaving sliding layers on plain
/// RoPE, so this is carried separately from `ropeTheta`/`fullRopeTheta`.
/// `nil` on an `ArchConfig` means no scaling, which is Gemma's behaviour.
public struct RopeScaling: Sendable, Equatable {
    public let factor: Double
    public let originalMaxPositionEmbeddings: Int
    public let betaFast: Double
    public let betaSlow: Double
    /// Multiplier on attention scores that accompanies the frequency scaling.
    public let attentionFactor: Double

    public init(factor: Double,
                originalMaxPositionEmbeddings: Int,
                betaFast: Double,
                betaSlow: Double,
                attentionFactor: Double) {
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.attentionFactor = attentionFactor
    }
}

/// Metal-compatible packed struct matching `RopeScalingParams` in Metal shaders.
public struct MetalRopeScalingParams: Sendable, Equatable {
    public var enabled: UInt32
    public var factor: Float
    public var originalMaxPositionEmbeddings: Float
    public var betaFast: Float
    public var betaSlow: Float

    public init(scaling: RopeScaling?) {
        if let s = scaling, s.factor > 1.0 {
            self.enabled = 1
            self.factor = Float(s.factor)
            self.originalMaxPositionEmbeddings = Float(s.originalMaxPositionEmbeddings)
            self.betaFast = Float(s.betaFast)
            self.betaSlow = Float(s.betaSlow)
        } else {
            self.enabled = 0
            self.factor = 1.0
            self.originalMaxPositionEmbeddings = 1.0
            self.betaFast = 1.0
            self.betaSlow = 1.0
        }
    }
}


/// Gating applied to the attention output before `o_proj`.
///
/// Laguna computes `gate = softplus(g_proj(hidden))` and multiplies the
/// attention output by it. `perHead` emits one gate per query head (broadcast
/// across `headDim`); `perElement` emits one per `(head, headDim)` channel.
/// Gemma has no such projection, hence `.none`.
public enum AttentionGating: String, Sendable, Equatable {
    case none
    case perHead
    case perElement
}

/// Architecture description for the loaded model. `manifest.json -> arch` must
/// match the config the runtime was handed, field-by-field, at load time;
/// mismatches throw `ModelError.archMismatch`.
///
/// Fields below the `tieWordEmbeddings` line describe variation that Gemma 4
/// does not exercise. They all default to Gemma's behaviour so existing
/// manifests and call sites are unaffected.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    /// Per-layer query-head count. Empty means every layer uses `numHeads`.
    /// Laguna varies it by layer type (48 on full-attention, 72 on sliding).
    public let headsPerLayer: [Int]
    /// 1 = the layer uses a plain dense MLP instead of routed experts, of
    /// width `denseMLPIntermediateSize`. Empty means every layer is sparse.
    /// Laguna marks layer 0 dense (`mlp_only_layers: [0]`).
    public let denseMLPLayerMask: [UInt8]
    /// FFN width of the dense layers selected by `denseMLPLayerMask`. This is
    /// the model's `intermediate_size`, which is distinct from the shared
    /// expert width carried in `intermediateSize` (Laguna: 12288 vs 1024).
    /// 0 when the model has no dense layers.
    public let denseMLPIntermediateSize: Int
    /// Partial rotary factor for full-attention layers when it differs from
    /// `partialRotaryFactor` (which then covers sliding layers only).
    /// `nil` means both layer types share `partialRotaryFactor`.
    public let fullPartialRotaryFactor: Double?
    /// YaRN scaling for full-attention layers. `nil` = plain RoPE.
    public let fullRopeScaling: RopeScaling?
    public let attentionGating: AttentionGating
    /// Multiplier on the routed-expert contribution (Laguna: 2.5).
    /// 1.0 is a no-op and matches Gemma.
    public let routedScalingFactor: Double

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        headsPerLayer: [Int] = [],
        denseMLPLayerMask: [UInt8] = [],
        denseMLPIntermediateSize: Int = 0,
        fullPartialRotaryFactor: Double? = nil,
        fullRopeScaling: RopeScaling? = nil,
        attentionGating: AttentionGating = .none,
        routedScalingFactor: Double = 1.0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.headsPerLayer = headsPerLayer
        self.denseMLPLayerMask = denseMLPLayerMask
        self.denseMLPIntermediateSize = denseMLPIntermediateSize
        self.fullPartialRotaryFactor = fullPartialRotaryFactor
        self.fullRopeScaling = fullRopeScaling
        self.attentionGating = attentionGating
        self.routedScalingFactor = routedScalingFactor
    }

    // MARK: - Per-layer accessors
    //
    // Every caller should go through these rather than reading `numHeads` or
    // `partialRotaryFactor` directly, so a model that varies them per layer
    // behaves correctly without touching each call site again.

    /// Query-head count for `layer`. Falls back to the uniform `numHeads`.
    public func numHeads(atLayer layer: Int) -> Int {
        guard layer >= 0, layer < headsPerLayer.count else { return numHeads }
        return headsPerLayer[layer]
    }

    /// Largest query-head count across all layers. Scratch buffers that must
    /// cover every layer size from one allocation should use this.
    public var maxNumHeads: Int {
        max(numHeads, headsPerLayer.max() ?? 0)
    }

    /// True when `layer` uses full attention rather than a sliding window.
    public func isFullAttention(layer: Int) -> Bool {
        guard layer >= 0, layer < fullAttentionLayerMask.count else { return false }
        return fullAttentionLayerMask[layer] == 1
    }

    /// True when `layer` uses a dense MLP instead of routed experts.
    public func isDenseMLP(layer: Int) -> Bool {
        guard layer >= 0, layer < denseMLPLayerMask.count else { return false }
        return denseMLPLayerMask[layer] == 1
    }

    /// True when `layer` uses a dense MLP instead of routed experts.
    public func isDenseMLP(atLayer layer: Int) -> Bool {
        return isDenseMLP(layer: layer)
    }

    /// Number of layers that actually carry routed experts. Expert streaming
    /// and packing should size themselves from this, not `numLayers`.
    public var numSparseLayers: Int {
        guard !denseMLPLayerMask.isEmpty else { return numLayers }
        return (0..<numLayers).reduce(0) { $0 + (isDenseMLP(layer: $1) ? 0 : 1) }
    }

    /// Partial rotary factor for `layer`, honouring a distinct full-attention
    /// value when the model specifies one.
    public func partialRotaryFactor(atLayer layer: Int) -> Double {
        guard isFullAttention(layer: layer), let full = fullPartialRotaryFactor else {
            return partialRotaryFactor
        }
        return full
    }

    /// RoPE base for `layer`.
    public func ropeTheta(atLayer layer: Int) -> Double {
        isFullAttention(layer: layer) ? fullRopeTheta : ropeTheta
    }

    /// YaRN scaling for `layer`, if the model scales that layer type.
    public func ropeScaling(atLayer layer: Int) -> RopeScaling? {
        isFullAttention(layer: layer) ? fullRopeScaling : nil
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// poolside/Laguna-S-2.1, transcribed from its `config.json`.
    ///
    /// This is the generalization target: it exercises every field Gemma does
    /// not — per-layer head counts, a dense layer 0, YaRN on full-attention
    /// layers only, per-head attention gating, top-10 routing, and a routed
    /// scaling factor. Not yet loadable; see the phase 0 work items.
    ///
    /// `intermediateSize` is the shared-expert width (1024); the dense layer-0
    /// FFN width (12288) is carried in `denseMLPIntermediateSize`.
    public static let lagunaS2_1 = ArchConfig(
        hiddenSize: 3072,
        intermediateSize: 1024,
        moeIntermediateSize: 1024,
        numHeads: 48,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 128,
        fullHeadDim: 128,
        vocabSize: 100352,
        slidingWindow: 512,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 500_000.0,
        partialRotaryFactor: 1.0,
        numLayers: 48,
        numExperts: 256,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.lagunaLayerMask(),
        hiddenActivation: "silu",
        headsPerLayer: Self.lagunaHeadsPerLayer(),
        denseMLPLayerMask: Self.lagunaDenseMask(),
        denseMLPIntermediateSize: 12288,
        fullPartialRotaryFactor: 0.5,
        fullRopeScaling: RopeScaling(factor: 128.0,
                                     originalMaxPositionEmbeddings: 8192,
                                     betaFast: 32.0,
                                     betaSlow: 1.0,
                                     attentionFactor: 1.4852030263919618),
        attentionGating: .perHead,
        routedScalingFactor: 2.5
    )

    /// Full attention every 4th layer starting at 0.
    private static func lagunaLayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 48)
        for i in stride(from: 0, to: 48, by: 4) { mask[i] = 1 }
        return mask
    }

    /// 48 query heads on full-attention layers, 72 on sliding layers.
    private static func lagunaHeadsPerLayer() -> [Int] {
        lagunaLayerMask().map { $0 == 1 ? 48 : 72 }
    }

    /// Only layer 0 is dense (`mlp_only_layers: [0]`).
    private static func lagunaDenseMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 48)
        mask[0] = 1
        return mask
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
