import Foundation
import Testing
@testable import TurboFieldfare

/// P3-2: delta rule recurrence + gating, plain Swift fp32.
///
/// Gate tests use hand-computed constants:
///   sigmoid(0) = 0.5; softplus(0) = ln(2) ≈ 0.693147
///   decay(aLog=0, a=0, dtBias=0) = exp(-1·ln2) = 0.5
///   decay(aLog=ln2, a=0, dtBias=0) = exp(-2·ln2) = 0.25
///
/// Scalar recurrence (Hv=1, Dk=Dv=1, q=k=v=1, beta=1, g=0.5):
///   t0: state 0→0, kv_mem=0, delta=1, state=1, y=1
///   t1: state 1→0.5, kv_mem=0.5, delta=0.5, state=1, y=1
///   t2: state 1→0.5, kv_mem=0.5, delta=0.5, state=1, y=1
///   ⇒ y=[1,1,1], final state 1 (invariant — write restores what decay took)
///
/// Full-sequence oracle constants come from `scripts/deltanet_oracle.py`,
/// a naive Python transcription cited in ADR-0002. Tolerance 1e-4 covers
/// fp32 (Swift) vs fp64 (Python) drift across three tokens.
@Suite struct DeltaNetRuleTests {

    // --- Gate tests (hand-derived). ---

    @Test func sigmoidAtZeroIsHalf() {
        #expect(abs(DeltaNetGate.sigmoid(0) - 0.5) < 1e-7)
    }

    @Test func softplusAtZeroIsLn2() {
        #expect(abs(DeltaNetGate.softplus(0) - Float(log(2.0))) < 1e-6)
    }

    @Test func softplusStableForLargeInput() {
        // For x > 20, softplus(x) ≈ x + exp(-x); at x=30 the correction is
        // ~1e-13, so fp32 should just return x.
        #expect(abs(DeltaNetGate.softplus(30) - 30) < 1e-5)
    }

    @Test func betaIsSigmoidOfInput() {
        let result = DeltaNetGate.beta([0, 100, -100])
        #expect(abs(result[0] - 0.5) < 1e-6)
        #expect(abs(result[1] - 1.0) < 1e-4)
        #expect(abs(result[2] - 0.0) < 1e-4)
    }

    @Test func decayAtZeroAlogYieldsHalf() {
        // exp(A_log)=1, softplus(0)=ln2, decay=exp(-ln2)=0.5.
        let result = DeltaNetGate.decay(a: [0], aLog: [0], dtBias: [0])
        #expect(abs(result[0] - 0.5) < 1e-6)
    }

    @Test func decayAtLn2AlogYieldsQuarter() {
        // exp(A_log)=2, softplus(0)=ln2, decay=exp(-2·ln2)=0.25.
        let result = DeltaNetGate.decay(a: [0], aLog: [Float(log(2.0))],
                                        dtBias: [0])
        #expect(abs(result[0] - 0.25) < 1e-6)
    }

    @Test func decayMatchesOracleG0() {
        // Oracle t0: A_log=[0, ln2], a=[0,0], dt_bias=[0,0.5].
        let a: [Float] = [0, 0]
        let aLog: [Float] = [0, Float(log(2.0))]
        let dtBias: [Float] = [0, 0.5]
        let g = DeltaNetGate.decay(a: a, aLog: aLog, dtBias: dtBias)
        // Oracle: g[0]=0.5, g[1]=0.14253695659655094.
        #expect(abs(g[0] - 0.5) < 1e-6)
        #expect(abs(g[1] - 0.14253695659655094) < 1e-4)
    }

    // --- Scalar recurrence walk (Hv=1, Dk=Dv=1). ---

    @Test func scalarRecurrenceSteadyState() {
        // q=k=v=1, beta=1, g=0.5 at every step. Hand derivation in header.
        var state = [Float](repeating: 0, count: 1)
        let expected: [Float] = [1, 1, 1]
        for step in 0..<3 {
            let y = DeltaNetRecurrence.step(
                q: [1], k: [1], v: [1], beta: [1], g: [0.5],
                numValueHeads: 1, headVDim: 1, headKDim: 1, state: &state)
            #expect(abs(y[0] - expected[step]) < 1e-6,
                    "step \(step): \(y[0]) vs \(expected[step])")
        }
        #expect(abs(state[0] - 1.0) < 1e-6)
    }

    @Test func scalarRecurrenceWithVaryingDecay() {
        // Varying g, q, k, v to exercise all the arithmetic.
        // Hand derivation per step (Hv=1, Dk=Dv=1):
        //   t0: g=0.5, decay 0→0, kv_mem=0, delta=(1-0)·1=1,
        //       state=0+1·1=1, y=1·1=1
        //   t1: g=0.25, decay 1→0.25, kv_mem=0.25·2=0.5,
        //       delta=(2-0.5)·0.5=0.75, state=0.25+2·0.75=1.75, y=1.75·1=1.75
        //   t2: g=0.1, decay 1.75→0.175, kv_mem=0.175·1=0.175,
        //       delta=(0.5-0.175)·1=0.325, state=0.175+1·0.325=0.5,
        //       y=0.5·2=1.0
        let g: [Float] = [0.5, 0.25, 0.1]
        let qs: [Float] = [1, 1, 2]
        let ks: [Float] = [1, 2, 1]
        let vs: [Float] = [1, 2, 0.5]
        let betas: [Float] = [1, 0.5, 1]
        let expected: [Float] = [1, 1.75, 1.0]
        var state = [Float](repeating: 0, count: 1)
        for step in 0..<3 {
            let y = DeltaNetRecurrence.step(
                q: [qs[step]], k: [ks[step]], v: [vs[step]],
                beta: [betas[step]], g: [g[step]],
                numValueHeads: 1, headVDim: 1, headKDim: 1, state: &state)
            #expect(abs(y[0] - expected[step]) < 1e-6,
                    "step \(step): \(y[0]) vs \(expected[step])")
        }
        #expect(abs(state[0] - 0.5) < 1e-6)
    }

    // --- QKNorm. ---

    @Test func qkNormScalesByInvDimAndInvSqrtDim() {
        // Dk=4, k=[1,1,1,1]. RMSNorm: mean_sq=1, inv=1, no change; then
        // scale by kScale(4)=1/sqrt(4)=0.5. Expected: [0.5, 0.5, 0.5, 0.5].
        var k: [Float] = [1, 1, 1, 1]
        DeltaNetQKNorm.applyInPlace(&k, numHeads: 1, headDim: 4,
                                    scale: DeltaNetQKNorm.kScale(headKDim: 4))
        for index in 0..<4 {
            #expect(abs(k[index] - 0.5) < 1e-5,
                    "index \(index): \(k[index])")
        }
    }

    @Test func qkNormMatchesOracleT0() {
        // Oracle t0: q=[[1,2]], k=[[1,0]], Dk=2, inv_scale=1/sqrt(2).
        // q_scale = 1/Dk = 0.5; k_scale = 1/sqrt(Dk) = 0.70710678...
        // q RMSNorm: mean_sq=(1+4)/2=2.5, inv=1/sqrt(2.5+1e-6)≈0.632455532;
        //   q_normed=[0.632455532, 1.264911064], ×0.5 → [0.316227766, 0.632455532].
        // Oracle: q_norm[0]=[0.31622770277130374, 0.6324554055426075] (fp64).
        var q: [Float] = [1, 2]
        DeltaNetQKNorm.applyInPlace(&q, numHeads: 1, headDim: 2,
                                    scale: DeltaNetQKNorm.qScale(headKDim: 2))
        #expect(abs(q[0] - 0.31622770277130374) < 1e-4)
        #expect(abs(q[1] - 0.6324554055426075) < 1e-4)

        var k: [Float] = [1, 0]
        DeltaNetQKNorm.applyInPlace(&k, numHeads: 1, headDim: 2,
                                    scale: DeltaNetQKNorm.kScale(headKDim: 2))
        // k RMSNorm: mean_sq=0.5, inv=1/sqrt(0.5+1e-6)≈1.41421; ×inv_scale
        //   → [1.0, 0.0]. Oracle: [0.9999990000015, 0.0] (eps perturbation).
        #expect(abs(k[0] - 0.9999990000015) < 1e-4)
        #expect(abs(k[1] - 0.0) < 1e-4)
    }

    // --- Head expansion. ---

    @Test func headExpansionIsRepeatInterleave() {
        // Hk=2, Hv=4, D=1: [1, 2] → [1, 1, 2, 2].
        let expanded = DeltaNetHeadExpansion.expand(
            [1, 2], numKeyHeads: 2, numValueHeads: 4, headDim: 1)
        #expect(expanded == [1, 1, 2, 2])
    }

    @Test func headExpansionWithMultiDimHeads() {
        // Hk=1, Hv=2, D=2: [[a,b]] → [[a,b],[a,b]].
        let expanded = DeltaNetHeadExpansion.expand(
            [3, 4], numKeyHeads: 1, numValueHeads: 2, headDim: 2)
        #expect(expanded == [3, 4, 3, 4])
    }

    @Test func headExpansionQwen36Shape() {
        // Hk=16, Hv=32, D=128: 16×128=2048 in, 32×128=4096 out.
        var input = [Float](repeating: 0, count: 16 * 128)
        for index in input.indices { input[index] = Float(index) }
        let expanded = DeltaNetHeadExpansion.expand(
            input, numKeyHeads: 16, numValueHeads: 32, headDim: 128)
        #expect(expanded.count == 32 * 128)
        // Value head h copies key head h/2:
        for head in 0..<32 {
            let sourceHead = head / 2
            for dim in 0..<128 {
                #expect(expanded[head * 128 + dim] == input[sourceHead * 128 + dim])
            }
        }
    }

    // --- Full oracle match (3-token sequence, Hk=1, Hv=2, Dk=Dv=2). ---

    @Test func fullSequenceMatchesOracle() {
        // Constants from scripts/deltanet_oracle.py. Tolerance 1e-4 covers
        // fp32 vs fp64 drift across 3 tokens.
        let Hk = 1, Hv = 2, Dk = 2, Dv = 2
        let aLog: [Float] = [0, Float(log(2.0))]
        let dtBias: [Float] = [0, 0.5]
        let normWeight: [Float] = [1, 0.5]

        struct Token { let q, k: [Float]; let v, z: [Float]; let a, b: [Float] }
        let tokens: [Token] = [
            // t0
            Token(q: [1, 2], k: [1, 0], v: [1, 0,  0, 1],
                  z: [1, -1,  0.5, 2], a: [0, 0], b: [0, 0]),
            // t1
            Token(q: [0.5, -1], k: [1, 1], v: [2, 0,  0, 0.5],
                  z: [0, 0,  1, 1], a: [0.25, -0.5], b: [1, -1]),
            // t2
            Token(q: [2, 1], k: [0, 1], v: [1, 1,  -1, 0],
                  z: [-2, 1,  0.25, -0.25], a: [1, 2], b: [0.5, 0]),
        ]

        // Oracle expected outputs (y = recurrence, out = gated).
        let expectedY: [[Float]] = [
            [0.15811369327203764, 0, 0, 0.15811369327203764],
            [-0.2324090114883663, 0, 0, 0.014775280353810934],
            [0.4269454341169863, 0.19683868753593547,
             -0.15811369327203764, 0.0008110323941780964],
        ]
        let expectedOut: [[Float]] = [
            [1.0338316042513689, 0, 0, 1.2455853508377372],
            [0, 0, 0, 0.514584712214845],
            [-0.3061813838692503, 0.21643233943877307,
             -0.19874884366707715, -0.0003969809917512999],
        ]
        let expectedFinalState: [Float] = [
            0.3154052135908391, 0.7193098512613327, 0, 0.6224587087434571,
            0, -0.49999950000075, 0.0011697453167567903, 0.00022521950121056144,
        ]

        var state = [Float](repeating: 0, count: Hv * Dv * Dk)
        let qScale = DeltaNetQKNorm.qScale(headKDim: Dk)
        let kScale = DeltaNetQKNorm.kScale(headKDim: Dk)

        for (step, token) in tokens.enumerated() {
            // Gating.
            let beta = DeltaNetGate.beta(token.b)
            let g = DeltaNetGate.decay(a: token.a, aLog: aLog, dtBias: dtBias)

            // QKNorm on key heads, then expand.
            var q = token.q
            var k = token.k
            DeltaNetQKNorm.applyInPlace(&q, numHeads: Hk, headDim: Dk,
                                        scale: qScale)
            DeltaNetQKNorm.applyInPlace(&k, numHeads: Hk, headDim: Dk,
                                        scale: kScale)
            let qExpanded = DeltaNetHeadExpansion.expand(
                q, numKeyHeads: Hk, numValueHeads: Hv, headDim: Dk)
            let kExpanded = DeltaNetHeadExpansion.expand(
                k, numKeyHeads: Hk, numValueHeads: Hv, headDim: Dk)

            // Recurrence.
            let y = DeltaNetRecurrence.step(
                q: qExpanded, k: kExpanded, v: token.v,
                beta: beta, g: g,
                numValueHeads: Hv, headVDim: Dv, headKDim: Dk,
                state: &state)

            for index in 0..<(Hv * Dv) {
                let expected = expectedY[step][index]
                #expect(abs(y[index] - expected) < 1e-4,
                        "t\(step) y[\(index)]: \(y[index]) vs \(expected)")
            }

            // Output gate.
            let out = DeltaNetOutputGate.apply(
                y: y, z: token.z, normWeight: normWeight,
                numValueHeads: Hv, headVDim: Dv)
            for index in 0..<(Hv * Dv) {
                let expected = expectedOut[step][index]
                #expect(abs(out[index] - expected) < 1e-4,
                        "t\(step) out[\(index)]: \(out[index]) vs \(expected)")
            }
        }

        for index in 0..<state.count {
            let expected = expectedFinalState[index]
            #expect(abs(state[index] - expected) < 1e-4,
                    "state[\(index)]: \(state[index]) vs \(expected)")
        }
    }

    // --- Real-dimension smoke test. ---

    @Test func qwen36ShapedRecurrenceProducesFiniteOutput() {
        let dims = DeltaNetDimensions.qwen36_35B_A3B
        let Hv = dims.numValueHeads
        let Dv = dims.headVDim
        let Dk = dims.headKDim

        var q = [Float](repeating: 0.01, count: Hv * Dk)
        var k = [Float](repeating: 0.01, count: Hv * Dk)
        let v = [Float](repeating: 0.01, count: Hv * Dv)
        let beta = [Float](repeating: 0.5, count: Hv)
        let g = [Float](repeating: 0.9, count: Hv)
        var state = [Float](repeating: 0, count: Hv * Dv * Dk)

        let y = DeltaNetRecurrence.step(
            q: q, k: k, v: v, beta: beta, g: g,
            numValueHeads: Hv, headVDim: Dv, headKDim: Dk, state: &state)

        #expect(y.count == Hv * Dv)
        #expect(y.allSatisfy { $0.isFinite })
        #expect(state.allSatisfy { $0.isFinite })

        // QKNorm should also not blow up at these dimensions.
        DeltaNetQKNorm.applyInPlace(&q, numHeads: Hv, headDim: Dk,
                                    scale: DeltaNetQKNorm.qScale(headKDim: Dk))
        DeltaNetQKNorm.applyInPlace(&k, numHeads: Hv, headDim: Dk,
                                    scale: DeltaNetQKNorm.kScale(headKDim: Dk))
        #expect(q.allSatisfy { $0.isFinite })
        #expect(k.allSatisfy { $0.isFinite })
    }
}
