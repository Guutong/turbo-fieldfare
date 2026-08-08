import Foundation

/// Gating math of Gated DeltaNet, plain Swift fp32 — the Phase 3 oracle form
/// (ADR-0001). Equations verified against ml-explore/mlx-lm `qwen3_5.py` /
/// `gated_delta.py` on 2026-08-08 (see PHASE-LOG P3-2 entry).
enum DeltaNetGate {
    static func sigmoid(_ x: Float) -> Float {
        1 / (1 + exp(-x))
    }

    /// log(1 + e^x), stable for large x (mlx `nn.softplus` = logaddexp(x, 0)).
    static func softplus(_ x: Float) -> Float {
        x > 20 ? x + log1p(exp(-x)) : log1p(exp(x))
    }

    /// beta = sigmoid(b), one entry per value head.
    static func beta(_ b: [Float]) -> [Float] {
        b.map(sigmoid)
    }

    /// g = exp(−exp(A_log) · softplus(a + dt_bias)), computed in fp32 with
    /// A_log upcast before the inner exp, exactly as `compute_g` in
    /// `gated_delta.py`.
    static func decay(a: [Float], aLog: [Float], dtBias: [Float]) -> [Float] {
        precondition(a.count == aLog.count && aLog.count == dtBias.count)
        return (0..<a.count).map { index in
            exp(-exp(aLog[index]) * softplus(a[index] + dtBias[index]))
        }
    }
}

/// Fixed-scale RMS normalization applied to q/k before the recurrence.
///
/// NOT a learned module and NOT an l2-norm: the source applies
/// `q = (inv_scale²) · rms_norm(q, None, 1e-6)` and
/// `k = inv_scale · rms_norm(k, None, 1e-6)` with `inv_scale = Dk^-0.5`,
/// per key head — net effect ‖k‖ ≈ 1, ‖q‖ ≈ 1/√Dk.
enum DeltaNetQKNorm {
    static let eps: Float = 1e-6

    /// In-place: each of `numHeads` rows of `headDim` elements becomes
    /// `scale · x · rsqrt(mean(x²) + eps)`.
    static func applyInPlace(_ x: inout [Float],
                             numHeads: Int,
                             headDim: Int,
                             scale: Float,
                             eps: Float = DeltaNetQKNorm.eps) {
        precondition(x.count == numHeads * headDim)
        for head in 0..<numHeads {
            let base = head * headDim
            var sumSquares: Float = 0
            for index in 0..<headDim {
                let value = x[base + index]
                sumSquares += value * value
            }
            let factor = scale / sqrt(sumSquares / Float(headDim) + eps)
            for index in 0..<headDim {
                x[base + index] *= factor
            }
        }
    }

    /// The q scale, `inv_scale² = 1/Dk`.
    static func qScale(headKDim: Int) -> Float {
        1 / Float(headKDim)
    }

    /// The k scale, `inv_scale = Dk^-0.5`.
    static func kScale(headKDim: Int) -> Float {
        1 / sqrt(Float(headKDim))
    }
}

/// 16 key heads → 32 value heads expansion.
enum DeltaNetHeadExpansion {
    /// `repeat_interleave`: value head `h` receives key head
    /// `h · numKeyHeads / numValueHeads` (i.e. `h // 2` for Qwen3.6). Both
    /// mlx paths (ops + Metal kernel) confirmed to use repeat, not tile.
    static func expand(_ x: [Float],
                       numKeyHeads: Int,
                       numValueHeads: Int,
                       headDim: Int) -> [Float] {
        precondition(x.count == numKeyHeads * headDim)
        precondition(numValueHeads % numKeyHeads == 0)
        var output = [Float](repeating: 0, count: numValueHeads * headDim)
        let repeatFactor = numValueHeads / numKeyHeads
        for head in 0..<numValueHeads {
            let source = (head / repeatFactor) * headDim
            let destination = head * headDim
            for index in 0..<headDim {
                output[destination + index] = x[source + index]
            }
        }
        return output
    }
}

/// The gated delta-rule recurrence, one decode timestep.
enum DeltaNetRecurrence {
    /// Updates `state` in place and returns this token's y.
    ///
    /// Ordering follows `_gated_delta_step_ops` / the Metal kernel exactly:
    /// decay state by g → kv_mem = Σ state·k → delta = (v − kv_mem)·beta →
    /// state += outer(k, delta) → y = Σ state·q. **Write-then-read**: y_t
    /// includes token t's own (k, v) write.
    ///
    /// - Parameters:
    ///   - q, k: `[numValueHeads × headKDim]`, head-expanded, already
    ///     RMS-normalized by `DeltaNetQKNorm`.
    ///   - v: `[numValueHeads × headVDim]`.
    ///   - beta, g: per value head, from `DeltaNetGate`.
    ///   - state: `[numValueHeads × headVDim × headKDim]` fp32.
    /// - Returns: y, `[numValueHeads × headVDim]`.
    static func step(q: [Float],
                     k: [Float],
                     v: [Float],
                     beta: [Float],
                     g: [Float],
                     numValueHeads: Int,
                     headVDim: Int,
                     headKDim: Int,
                     state: inout [Float]) -> [Float] {
        precondition(q.count == numValueHeads * headKDim)
        precondition(k.count == numValueHeads * headKDim)
        precondition(v.count == numValueHeads * headVDim)
        precondition(beta.count == numValueHeads && g.count == numValueHeads)
        precondition(state.count == numValueHeads * headVDim * headKDim)

        var y = [Float](repeating: 0, count: numValueHeads * headVDim)
        for head in 0..<numValueHeads {
            let decay = g[head]
            let writeScale = beta[head]
            let kBase = head * headKDim
            let vBase = head * headVDim
            for valueIndex in 0..<headVDim {
                let stateBase = (head * headVDim + valueIndex) * headKDim
                var kvMemory: Float = 0
                for index in 0..<headKDim {
                    state[stateBase + index] *= decay
                    kvMemory += state[stateBase + index] * k[kBase + index]
                }
                let delta = (v[vBase + valueIndex] - kvMemory) * writeScale
                var accumulator: Float = 0
                for index in 0..<headKDim {
                    state[stateBase + index] += k[kBase + index] * delta
                    accumulator += state[stateBase + index] * q[kBase + index]
                }
                y[vBase + valueIndex] = accumulator
            }
        }
        return y
    }
}

/// Post-recurrence output gate: `out = silu(z) · rms_norm(y, weight)`,
/// computed in fp32 (mlx `_precise_swiglu`), per value-head row.
enum DeltaNetOutputGate {
    /// - Parameters:
    ///   - y: recurrence output, `[numValueHeads × headVDim]`.
    ///   - z: `in_proj_z` output, same shape.
    ///   - normWeight: the learned `linear_attn.norm.weight`, `headVDim`.
    ///   - eps: the layer's rms epsilon (config `rms_norm_eps`, 1e-6 here).
    static func apply(y: [Float],
                      z: [Float],
                      normWeight: [Float],
                      numValueHeads: Int,
                      headVDim: Int,
                      eps: Float = 1e-6) -> [Float] {
        precondition(y.count == numValueHeads * headVDim)
        precondition(z.count == y.count)
        precondition(normWeight.count == headVDim)

        var output = [Float](repeating: 0, count: y.count)
        for head in 0..<numValueHeads {
            let base = head * headVDim
            var sumSquares: Float = 0
            for index in 0..<headVDim {
                let value = y[base + index]
                sumSquares += value * value
            }
            let invRms = 1 / sqrt(sumSquares / Float(headVDim) + eps)
            for index in 0..<headVDim {
                let gate = z[base + index]
                let siluGate = gate / (1 + exp(-gate))
                output[base + index] =
                    siluGate * y[base + index] * normWeight[index] * invRms
            }
        }
        return output
    }
}
