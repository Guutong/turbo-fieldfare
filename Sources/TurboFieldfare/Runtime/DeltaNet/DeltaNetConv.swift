import Foundation

/// Causal depthwise conv1d (width 4) over the fused qkv stream, plain Swift
/// fp32 — the Phase 3 oracle form (ADR-0001).
///
/// Mirrors mlx's `GatedDeltaNet` decode step: the kernel is applied as
/// cross-correlation over `concat(conv_state, qkv)`, so for token `t`
///
///     out[c] = silu( w[c,0]·x[t-3] + w[c,1]·x[t-2] + w[c,2]·x[t-1]
///                    + w[c,3]·x[t] )
///
/// with `x[<0]` taken from the conv state (zeros at session start) and the
/// silu matching the source's post-conv activation.
enum DeltaNetConv {
    /// One decode step.
    ///
    /// - Parameters:
    ///   - input: this token's pre-conv qkv vector, `convDim` elements.
    ///   - weight: per-channel taps, row-major `[convDim][4]` — the
    ///     `(8192, 4, 1)` checkpoint weight with tap 0 = oldest input,
    ///     matching cross-correlation over `concat(conv_state, qkv)`.
    ///   - convDim: width of the fused qkv stream (8192 for Qwen3.6).
    ///   - convState: in-out, `[3 × convDim]` rows oldest-first; on return it
    ///     holds the three most recent inputs, still oldest-first.
    /// - Returns: the silu'd conv output, `convDim` elements.
    static func step(input: [Float],
                     weight: [Float],
                     convDim: Int,
                     convState: inout [Float]) -> [Float] {
        precondition(input.count == convDim, "input must have convDim elements")
        precondition(weight.count == convDim * 4,
                     "weight must be convDim × 4 taps")
        precondition(convState.count == convDim * 3,
                     "conv state must hold 3 × convDim elements")

        var output = [Float](repeating: 0, count: convDim)
        for channel in 0..<convDim {
            let tap = channel * 4
            let accumulator =
                weight[tap + 0] * convState[channel]
                + weight[tap + 1] * convState[convDim + channel]
                + weight[tap + 2] * convState[2 * convDim + channel]
                + weight[tap + 3] * input[channel]
            output[channel] = silu(accumulator)
        }

        // Slide the window: drop the oldest row, append this token.
        for row in 0..<2 {
            for channel in 0..<convDim {
                convState[row * convDim + channel] =
                    convState[(row + 1) * convDim + channel]
            }
        }
        for channel in 0..<convDim {
            convState[2 * convDim + channel] = input[channel]
        }
        return output
    }

    /// silu(x) = x / (1 + e^-x), matching the activation the source applies
    /// after the conv.
    @inline(__always)
    static func silu(_ x: Float) -> Float {
        x / (1 + exp(-x))
    }
}
