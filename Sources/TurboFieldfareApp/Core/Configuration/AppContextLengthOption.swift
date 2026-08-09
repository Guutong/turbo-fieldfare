import TurboFieldfare

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    public var fp16KVBytes: UInt64 {
        fp16KVBytes(for: .gemma4_26B_A4B)
    }

    /// KV-cache footprint at this context length for a given architecture.
    /// Sliding layers only ever hold `slidingWindow` rows plus one prefill
    /// chunk; full-attention layers hold the whole context.
    public func fp16KVBytes(for architecture: ArchConfig) -> UInt64 {
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        let slidingLayers = architecture.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var menuLabel: String {
        switch self {
        case .fourK: "4K, Default"
        case .eightK: "8K, +85 MB"
        case .sixteenK: "16K, +250 MB"
        }
    }
}
