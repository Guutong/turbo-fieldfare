import Foundation
import Metal

/// Results from a completed batched forward pass.
internal struct BatchedPassResult: Sendable {
    /// Greedy argmax token ID for each of K draft positions.
    let greedyTokens: [UInt32]
    /// Number of accepted drafts before first mismatch (<= K).
    var acceptedCount: Int = 0
}

/// Orchestrates a forward pass for K draft tokens.
///
/// Process order: for each draft token k (0..K-1), run all N layers
/// end-to-end before moving to k+1. This matches the dependency chain:
///   - Token-k's QKV/Auxiliary outputs depend on token-k's previous-layer residual
///   - DeltaNet recurrence is inherently sequential within one token
///   - KV cache position advances with each draft token
///
/// Layer-major full-batching (all K tokens' layer-L ops in one CB) would
/// require new kernel variants that accept per-token stride arrays. Most
/// fused kernels (FusedQKVGEMV, FusedPostAttnSetup) currently operate on
/// a single-row at fixed offset 0.
enum DraftVerifier {

    // MARK: – Public entry point

    /// Run a forward pass for K draft tokens starting at the given KV cache
    /// position. Returns the greedy argmax for each position along with the
    /// number of previously accepted consecutive drafts.
    ///
    /// - Parameters:
    ///   - runner: The RealForwardRunner executing this pass.
    ///   - draftTokens: K draft token IDs to verify.
    ///   - startKVPosition: KV cache position where token 0 begins.
    ///   - logProbScratchOffset: Unused in current impl (fused greedy head).
    /// - Returns: Result with `greedyTokens[k]` = model's greedy output at
    ///   draft position k.
    static func run(
        _ runner: RealForwardRunner,
        draftTokens: [Int32],
        startKVPosition: Int,
        _ logProbScratchOffset: UInt = 0
    ) async throws -> BatchedPassResult {
        let K = draftTokens.count
        guard K > 0 else {
            return BatchedPassResult(greedyTokens: [])
        }

        let ctx       = runner.ctx
        let cfg       = runner.cfg
        let model     = runner.model
        let device    = ctx.device
        let queue     = ctx.queue
        let eps: Float = 1e-6
        let D         = UInt32(cfg.hiddenSize)

        // Scratch buffers were allocated in the runner's init with
        // batchSize = scratchSize(for: D) which already covers K >= 1.
        let hidden       = runner.hidden
        let attnOut      = runner.attnOut
        let qScratch     = runner.qScratch
        let qRawScratch  = runner.qRawScratch
        let denseX       = runner.denseX
        let routedX      = runner.routedX
        let routerInput  = runner.routerInput
        let outIndices   = runner.outIndices
        let outWeights   = runner.outWeights
        let logitsBytes = Int(cfg.vocabSize) * MemoryLayout<Float>.size
        guard let logitsBuf = device.makeBuffer(
                length: K * logitsBytes, options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }

        // Per-token scratch: temporary normed buffer for head computation.
        guard let tmpNormed = device.makeBuffer(
                length: Int(D) * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }

        // ── Embedding (shared across all draft tokens — done once) ────────
        let emb = model.embedding
        let sqrtHidden: Float = cfg.topology == .qwen36
            ? 1 : Float(D).squareRoot()

        do {
            let cb = queue.makeCommandBuffer()!
            for k in 0..<K {
                let tokenId = UInt32(bitPattern: draftTokens[k])
                runner.embedInt4.encode(commandBuffer: cb,
                                        table: emb.buffer, tableOffset: Int(emb.offset),
                                        scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                        biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                        out: hidden,
                                        outOffset: k * Int(D) * MemoryLayout<Float16>.stride,
                                        tokenId: tokenId,
                                        d: D,
                                        outScale: sqrtHidden)
            }
            cb.commit()
            waitForCommandBuffer(cb)
            try checkCmdError(cb.error)
        }

        // ── Main loop: layer-major for reduced MoE sync overhead ─────
        //
        // Originally token-major (process all N layers per tk, then advance).
        // Layer-major (process K tokens per layer, drain MoE once/layer)
        // saves ~4*K sync operations per middle layer (commit, waitForCB,
        // checkErr, unpinSlots) — the biggest win because finishPendingMoE
        // has per-call Metal bridge overhead that dominates at small K.
        var greedyTokens: [UInt32] = []
        greedyTokens.reserveCapacity(K)
        var layerPlans: [(layer: Int, experts: [Int])] = []

        for tk in 0..<K {
            let seqLen = UInt32(startKVPosition + tk + 1)
            let tokenOff = tk * Int(D) * MemoryLayout<Float16>.stride

            for L in 0..<cfg.numLayers {
                let isFull   = cfg.layerKindMask[L] == 1
                let isLinear = cfg.layerKindMask[L] == 2 // DeltaNet
                let isDense  = cfg.isDenseMLP(atLayer: L)
                let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
                let numKVL   = isFull
                    ? max(cfg.numFullKVHeads, cfg.numKVHeads)
                    : cfg.numKVHeads
                let qDim = UInt32(cfg.numHeads * headDimL)
                let kvDim = UInt32(numKVL * headDimL)

                // ── Resolve per-layer tensors ────────────────────────
                let inNorm   = try model.inputNorm(layer: L)
                let postAttn = try model.postAttnNorm(layer: L)
                let q: TensorView
                let k: TensorView
                let vProj: TensorView
                let o: TensorView
                if isLinear {
                    q = postAttn; k = postAttn; vProj = postAttn; o = postAttn
                } else {
                    q    = try model.qProj(layer: L)
                    k    = try model.kProj(layer: L)
                    vProj = (isFull && cfg.attentionKEqV)
                                ? k : (try model.vProj(layer: L))
                    o    = try model.oProj(layer: L)
                }
                let qNormT: TensorView
                let kNormT: TensorView
                if isLinear {
                    qNormT = postAttn; kNormT = postAttn
                } else {
                    qNormT = try model.qNorm(layer: L)
                    kNormT = try model.kNorm(layer: L)
                }
                let preFFN  = cfg.topology == .qwen36 ? postAttn
                                : (try model.preFFN(layer: L))
                let preFFN2 = cfg.topology == .qwen36 ? postAttn
                                : (try model.preFFN2(layer: L))
                let _ = cfg.topology == .qwen36 ? postAttn
                           : (try model.postFFN2(layer: L)) // tail norm
                let _ = cfg.topology == .qwen36 ? postAttn
                           : (try model.postFFN(layer: L))  // residual combine
                let _ = runner.sharedExpertProjections[L] // for deferred MoE
                let routerW: TensorView? = isDense ? nil
                                   : (try model.router(layer: L))
                let selectionBias = isDense
                    ? nil : (try model.routerSelectionBias(layer: L))

                let isGatedAttn = cfg.topology == .qwen36 && isFull
                let _ = try model.layerScalar(layer: L)
                    .map { Quantization.bf16ToFloat(
                        $0.buffer.contents().advanced(
                            by: Int($0.offset)).assumingMemoryBound(
                                to: UInt16.self)[0]) } ?? 1.0

                // ── Drain pending MoE BEFORE emitting layer compute ────
                if !layerPlans.isEmpty {
                    try finishPendingMoE(runner, layerPlans, L, waitIfNeeded: true)
                    layerPlans.removeAll()
                }

                // ── DeltaNet branch — batch-in-time via encodeBatched ────
                if isLinear {
                    guard let deltaNet = runner.deltaNet,
                          let weightsCache = runner.deltaNetWeights,
                          let stateStore = runner.deltaNetState,
                          let convState = stateStore.convState[L],
                          let recurrentState = stateStore.recurrentState[L]
                    else {
                        preconditionFailure(
                            "qwen36 topology requires DeltaNet state/weights")
                    }
                    // encodeBatched(tk+1) processes tokens 0..tk in a single
                    // Metal command buffer per tk.  Each call recomputes all
                    // previously seen tokens from their original inputs (same
                    // semantic as the sequential encode()-loop above) but
                    // collapses O(K²) Metal CB syncs down to O(K).
                    // Result folds into hidden at per-token offsets via
                    // store_hidden_batched, so no post-copy is needed.
                    let cb = queue.makeCommandBuffer()!
                    deltaNet.encodeBatched(commandBuffer: cb,
                                           hidden: hidden,
                                           weights: try weightsCache.weights(
                                               layer: L),
                                           convState: convState,
                                           recurrentState: recurrentState,
                                           tk: tk + 1,
                                           eps: eps)
                    cb.commit()
                    waitForCommandBuffer(cb)
                    try checkCmdError(cb.error)
                } else {
                    // ── Full / SWA attention path ────────────────────────
                    var cb = queue.makeCommandBuffer()!

                    // A) RMSNorm input — reads AND writes hidden
                    runner.rms.encodeBF16W(
                        commandBuffer: cb,
                        x: hidden,
                        xOffset: tokenOff,
                        weight: inNorm.buffer,
                        weightOffset: Int(inNorm.offset),
                        out: hidden,
                        outOffset: tokenOff,
                        d: D, eps: eps)

                    // B) Fused QKV GEMV — data buffers hardcoded at offset 0
                    // Copy token tk from its slot to offset-0 region first.
                    let attnQuant = runner.attentionQuantByLayer[L]
                    let kSlot = runner.kv?.kSlot(
                                    layer: L,
                                    position: startKVPosition + tk)
                                ?? (buffer: runner.kStage, offset: 0)
                    let vSlot = runner.kv?.vSlot(
                                    layer: L,
                                    position: startKVPosition + tk)
                                ?? (buffer: runner.vStage, offset: 0)

                    // Copy token tk's hidden to offset 0 for kernel call.
                    copyScratchRegion(from: hidden,
                                      srcStart: tokenOff,
                                      dstStart: 0,
                                      byteCount: Int(D)
                                              * MemoryLayout<Float16>
                                                      .stride)

                    if attnQuant.weightBits != 4 {
                        // Wide GEMV — also lacks xOffset. Same fix applies.
                        guard let wide = runner.fusedQKVGEMVWide[
                                attnQuant.weightBits] else {
                            preconditionFailure(
                                "no QKV pipeline for \(attnQuant.weightBits)-bit"
                                + " attention at layer \(L)")
                        }
                        wide.encode(commandBuffer: cb,
                            qWeights: q.buffer,
                            qWeightsOffset: Int(q.offset),
                            qScales: q.buffer,
                            qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer,
                            qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer,
                            kWeightsOffset: Int(k.offset),
                            kScales: k.buffer,
                            kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer,
                            kBiasesOffset: Int(k.biasOffset),
                            vWeights: vProj.buffer,
                            vWeightsOffset: Int(vProj.offset),
                            vScales: vProj.buffer,
                            vScalesOffset: Int(vProj.scaleOffset),
                            vBiases: vProj.buffer,
                            vBiasesOffset: Int(vProj.biasOffset),
                            // NOTE: No xOffset — kernel hardcodes 0
                            x: hidden,
                            qOut: qScratch,
                            kOut: kSlot.buffer,
                            kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer,
                            vOutOffset: vSlot.offset,
                            qRows: qDim,
                            kvRows: kvDim,
                            n: UInt32(D),
                            groupSize: UInt32(
                                attnQuant.groupSize))
                    } else {
                        runner.fusedQKVGEMV.encode(
                            commandBuffer: cb,
                            qWeights: q.buffer,
                            qWeightsOffset: Int(q.offset),
                            qScales: q.buffer,
                            qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer,
                            qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer,
                            kWeightsOffset: Int(k.offset),
                            kScales: k.buffer,
                            kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer,
                            kBiasesOffset: Int(k.biasOffset),
                            vWeights: vProj.buffer,
                            vWeightsOffset: Int(vProj.offset),
                            vScales: vProj.buffer,
                            vScalesOffset: Int(vProj.scaleOffset),
                            vBiases: vProj.buffer,
                            vBiasesOffset: Int(vProj.biasOffset),
                            // NOTE: No xOffset — kernel hardcodes 0
                            x: hidden,
                            qOut: isGatedAttn
                                ? qRawScratch : qScratch,
                            kOut: kSlot.buffer,
                            kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer,
                            vOutOffset: vSlot.offset,
                            qRows: isGatedAttn ? 2 * qDim : qDim,
                            kvRows: kvDim,
                            n: D)
                    }
                    cb.commit()

                    // C) Handle gated-Q de-interleave + epilogue
                    if isGatedAttn {
                        waitForCommandBuffer(cb)
                        try checkCmdError(cb.error)
                        runner.splitGatedQProjection(
                            qDim: Int(qDim),
                            headDim: headDimL)

                        cb = queue.makeCommandBuffer()!

                        // RoPE + norm on Q; K/V stay at their cache slots.
                        let rotated: UInt32 = isFull
                            ? UInt32(
                                Double(cfg.fullHeadDim)
                                * cfg.partialRotaryFactor / 2.0)
                            : UInt32(headDimL / 2)
                        let scaling = cfg.ropeScaling(atLayer: L)

                        runner.fusedQKVEpilogue.encode(
                            commandBuffer: cb,
                            q: qScratch, qOffset: 0,
                            k: kSlot.buffer, kOffset: kSlot.offset,
                            v: vSlot.buffer, vOffset: vSlot.offset,
                            qWeight: qNormT.buffer,
                            qWeightOffset: Int(qNormT.offset),
                            kWeight: kNormT.buffer,
                            kWeightOffset: Int(kNormT.offset),
                            headDim: UInt32(headDimL),
                            numQHeads: UInt32(cfg.numHeads),
                            numKVHeads: UInt32(numKVL),
                            position: UInt32(startKVPosition + tk),
                            theta: isFull
                                ? Float(cfg.fullRopeTheta)
                                : Float(cfg.ropeTheta),
                            rotatedPairs: rotated,
                            eps: eps,
                            ropePairStride: cfg.topology == .qwen36
                                ? rotated : nil,
                            ropeFreqDim: cfg.topology == .qwen36
                                ? 2 * rotated : nil,
                            normalizeV: cfg.topology != .qwen36,
                            scaling: scaling)
                        cb.commit()

                        waitForCommandBuffer(cb)
                        try checkCmdError(cb.error)
                        runner.applyAttentionOutputGate(qDim: Int(qDim))

                        cb = queue.makeCommandBuffer()!
                    }

                    // D) Attention kernel — data at offset 0 (already copied)
                    let attnScale: Float = isFull
                        ? Float(cfg.ropeScaling(atLayer: L)?
                                .attentionFactor ?? 1.0)
                        : 1.0

                    if isFull {
                        runner.attention.encodeFull(
                            commandBuffer: cb,
                            q: isGatedAttn ? qScratch : qRawScratch,
                            qOffset: 0,
                            k: kSlot.buffer, kOffset: 0,
                            v: vSlot.buffer, vOffset: 0,
                            out: attnOut, outOffset: 0,
                            headDim: UInt32(headDimL),
                            numQHeads: UInt32(cfg.numHeads),
                            numKVHeads: UInt32(numKVL),
                            seqLen: seqLen,
                            scale: cfg.topology == .qwen36
                                ? 1 / Float(headDimL).squareRoot()
                                : attnScale)
                    } else {
                        let ringCapacity = runner.kv?
                            .ringCapacity(layer: L) ?? 0
                        let activeRC = ringCapacity > 0
                            && Int(seqLen) > ringCapacity
                            ? UInt32(ringCapacity) : 0
                        runner.attention.encodeSWA(
                            commandBuffer: cb,
                            q: isGatedAttn ? qScratch : qScratch,
                            qOffset: 0,
                            k: kSlot.buffer, kOffset: 0,
                            v: vSlot.buffer, vOffset: 0,
                            out: attnOut, outOffset: 0,
                            headDim: UInt32(headDimL),
                            numQHeads: UInt32(cfg.numHeads),
                            numKVHeads: UInt32(numKVL),
                            seqLen: seqLen,
                            window: UInt32(cfg.slidingWindow),
                            scale: 1.0,
                            ringCapacity: activeRC)
                    }
                    cb.commit()

                    // E) o_proj — data already at offset 0
                    cb = queue.makeCommandBuffer()!
                    runner.int4.encode(commandBuffer: cb,
                                       weights: o.buffer,
                                       weightsOffset: Int(o.offset),
                                       scales: o.buffer,
                                       scalesOffset: Int(o.scaleOffset),
                                       biases: o.buffer,
                                       biasesOffset: Int(o.biasOffset),
                                       x: attnOut, xOffset: 0,
                                       y: attnOut, yOffset: 0,
                                       m: D, n: qDim)

                    // F) Post-attention setup (residual + norm)
                    // NOTE: FusedPostAttentionSetup has ONLY weight-offsets.
                    // Data buffers (index 0-4) are hardcoded at offset 0.
                    // We already copied token tk's hidden/oOut to offset 0
                    // above, so we can call the non-fused elements directly.
                    if cfg.topology == .qwen36 {
                        if !isLinear {
                            runner.elementwiseAdd.encode(
                                commandBuffer: cb,
                                a: hidden, aOffset: 0,
                                b: attnOut, bOffset: 0,
                                count: D)
                        }
                        // Norm on hidden -> denseX (router input)
                        runner.rms.encodeBF16W(
                            commandBuffer: cb,
                            x: hidden, xOffset: 0,
                            weight: postAttn.buffer,
                            weightOffset: Int(postAttn.offset),
                            out: denseX, outOffset: 0,
                            d: D, eps: eps)
                    } else {
                        switch cfg.normTopology {
                        case .preNorm:
                            // Non-Qwen3.6 pre-norm path
                            runner.fusedPostAttentionSetup.encodePreNorm(
                                commandBuffer: cb,
                                hidden: hidden, attn: attnOut,
                                denseX: denseX, routedX: routedX,
                                routerX: routerInput,
                                preFFNWeight: preFFN.buffer,
                                preFFNWeightOffset: Int(preFFN.offset),
                                d: D, eps: eps)
                        case .sandwich:
                            runner.fusedPostAttentionSetup.encode(
                                commandBuffer: cb,
                                hidden: hidden, attn: attnOut,
                                denseX: denseX, routedX: routedX,
                                routerX: routerInput,
                                postAttentionWeight: postAttn.buffer,
                                postAttentionWeightOffset: Int(
                                    postAttn.offset),
                                preFFNWeight: preFFN.buffer,
                                preFFNWeightOffset: Int(preFFN.offset),
                                preFFN2Weight: preFFN2.buffer,
                                preFFN2WeightOffset: Int(preFFN2.offset),
                                d: D, eps: eps)
                        default:
                            break
                        }
                    }
                    cb.commit()
                    waitForCommandBuffer(cb)
                    try checkCmdError(cb.error)
                }

                // ── Router (collect plan for deferred MoE) ─────────────
                if !isDense, let routerW = routerW {
                    let effectiveScaleBuf = runner.effectiveScaleBuffers[L]
                    let cb = queue.makeCommandBuffer()!
                    let selBias = selectionBias

                    if cfg.routerScoring == .sigmoidTopK,
                       let sb = selBias {
                        runner.moe.encodeRouterSigmoid(
                            commandBuffer: cb,
                            weights: routerW.buffer,
                            weightsOffset: Int(routerW.offset),
                            scales: routerW.buffer,
                            scalesOffset: Int(routerW.scaleOffset),
                            biases: routerW.buffer,
                            biasesOffset: Int(routerW.biasOffset),
                            hidden: denseX,
                            effectiveScale: effectiveScaleBuf,
                            perExpertScale: runner.routerOnesPerExpert,
                            selectionBias: sb.buffer,
                            selectionBiasOffset: Int(sb.offset),
                            outIndices: outIndices,
                            outWeights: outWeights,
                            numExperts: UInt32(cfg.numExperts),
                            d: D,
                            topK: UInt32(cfg.topKExperts),
                            groupSize: UInt32(model.routerGroupSize),
                            useBF16: routerW.scaleLength == 0)
                    } else {
                        _ = runner.sharedExpertProjections[L] // for deferred MoE
                        runner.moe.encodeRouterGemma4(
                            commandBuffer: cb,
                            weights: routerW.buffer,
                            weightsOffset: Int(routerW.offset),
                            scales: routerW.buffer,
                            scalesOffset: Int(routerW.scaleOffset),
                            biases: routerW.buffer,
                            biasesOffset: Int(routerW.biasOffset),
                            hidden: denseX,
                            effectiveScale: effectiveScaleBuf,
                            perExpertScale: runner.routerOnesPerExpert,
                            outIndices: outIndices,
                            outWeights: outWeights,
                            numExperts: UInt32(cfg.numExperts),
                            d: D,
                            topK: UInt32(cfg.topKExperts),
                            groupSize: UInt32(model.routerGroupSize),
                            useBF16: routerW.scaleLength == 0)
                    }
                    cb.commit()
                    waitForCommandBuffer(cb)
                    try checkCmdError(cb.error)

                    // Read outIndices for this token's routing decision.
                    let idxPtr = outIndices.contents()
                        .bindMemory(to: UInt32.self,
                                    capacity: cfg.topKExperts)
                    var experts = [Int](repeating: 0,
                                        count: cfg.topKExperts)
                    for i in 0..<cfg.topKExperts {
                        experts[i] = min(Int(idxPtr[i]),
                                         cfg.numExperts - 1)
                    }
                    layerPlans.append((layer: L, experts: experts))
                }
            } // end layer loop

            // ── Drain remaining pending MoE ───────────────────────────────
            if !layerPlans.isEmpty {
                try finishPendingMoE(runner, layerPlans, 0, waitIfNeeded: true)
                layerPlans.removeAll()
            }

            // ── Final norm + lm_head + argmax for this token ─────────────
            let fNorm = model.finalNorm
            let lm = model.lmHead
            let logitBase = tk * Int(cfg.vocabSize)
                           * MemoryLayout<Float>.size

            do {
                let finalCB = queue.makeCommandBuffer()!
                runner.rms.encodeBF16W(commandBuffer: finalCB,
                                       x: hidden,
                                       xOffset: tokenOff,
                                       weight: fNorm.buffer,
                                       weightOffset: Int(fNorm.offset),
                                       out: tmpNormed,
                                       d: D, eps: eps)

                runner.int4.encode(commandBuffer: finalCB,
                                   weights: lm.buffer,
                                   weightsOffset: Int(lm.offset),
                                   scales: lm.buffer,
                                   scalesOffset: Int(lm.scaleOffset),
                                   biases: lm.buffer,
                                   biasesOffset: Int(lm.biasOffset),
                                   x: tmpNormed,
                                   y: logitsBuf,
                                   yOffset: logitBase,
                                   m: UInt32(cfg.vocabSize), n: D)
                finalCB.commit()
                waitForCommandBuffer(finalCB)
                try checkCmdError(finalCB.error)
            }

            // Argmax
            let logitsPtr = logitsBuf.contents()
                .assumingMemoryBound(to: Float.self)
            let base = logitBase / MemoryLayout<Float>.stride
            var bestIdx: Int32 = 0
            var bestVal: Float = -.infinity
            for v in 0..<Int(cfg.vocabSize) {
                let val = logitsPtr[base + v]
                if val > bestVal {
                    bestVal = val
                    bestIdx = Int32(v)
                }
            }
            greedyTokens.append(UInt32(bitPattern: bestIdx))
        } // end draft-token loop

        // Compute accepted count: longest prefix of draftTokens that match
        // the model's own greedy outputs.
        var accepted = 0
        for k in 0..<greedyTokens.count {
            if greedyTokens[k] == UInt32(bitPattern: draftTokens[k]) {
                accepted += 1
            } else {
                break
            }
        }

        return BatchedPassResult(greedyTokens: greedyTokens,
                                 acceptedCount: accepted)
    }

    // MARK: – Helpers

    /// Synchronously wait for a command buffer to complete (async-compatible).
    /// Uses the underlying Objective-C setCompletionHandler since `any
    /// MTLCommandBuffer` doesn't expose Metal completion methods.
    private static func waitForCommandBuffer(_ cb: MTLCommandBuffer) {
        let semaphore = DispatchSemaphore(value: 0)
        typealias HandlerBlock = @convention(block) (MTLCommandBuffer?) -> Void
        var capturedSem = semaphore
        let block: HandlerBlock = { _ in capturedSem.signal() }
        let obj = unsafeBitCast(block, to: AnyObject.self)
        (cb as AnyObject).perform(Selector(("addCompletedHandler:")), with: obj)
        semaphore.wait()
    }

    /// Check command buffer status and throw on failure.
    private static func checkCmdError(_ error: Error?) throws {
        if let error { throw error }
    }

    /// Copy a contiguous byte range within a single Metal buffer.
    private static func copyScratchRegion(
        from buf: MTLBuffer,
        srcStart: Int,
        dstStart: Int,
        byteCount: Int
    ) {
        let src = buf.contents().advanced(by: srcStart)
        let dst = buf.contents().advanced(by: dstStart)
        memcpy(dst, src, byteCount)
    }

    // MARK: – Deferred MoE plumbing

    /// Execute a batch of deferred MoE operations (one per layer in `plans`).
    ///
    /// This is the DraftVerifier's copy of produceToken's MoE execution pipeline.
    /// For each layer that was routed in the draft phase:
    ///   1. Plan expert cache fetch (hits vs misses)
    ///   2. Pin cache slots to prevent eviction during GPU work
    ///   3. Shared expert FFN → h1Buf via Metal command buffer
    ///   4. Apply sigmoid gate scalar on CPU (Qwen3.6 only)
    ///   5. Async disk fetch for expert misses → TensorViews
    ///   6. Phase‑1 load+activation into moeActs
    ///   7. Phase‑2 down‑projection + reduce/scatter → h2Buf
    ///   8. Tail kernel: hidden + h1 + routedScale*h2 (preNorm topology)
    ///   9. Sync wait + unpin slots
    ///
    /// For Qwen3.6 topology (`qwen3_5_moe`) every layer has routed experts
    /// so `plans` always contains entries. For non-MoE topologies it may be
    /// empty (no-op).
    private static func finishPendingMoE(
        _ runner: RealForwardRunner,
        _ plans: [(layer: Int, experts: [Int])],
        _ currentLayer: Int,
        waitIfNeeded: Bool
    ) throws {
        guard !plans.isEmpty else { return }

        let queue = runner.ctx.queue
        let cfg  = runner.cfg
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let topK = UInt32(cfg.topKExperts)
        let isQwen36 = cfg.topology == .qwen36

        for planTuple in plans {
            let L     = planTuple.layer
            let experts = planTuple.experts
            let denseX = runner.denseX
            let sharedProj = runner.sharedExpertProjections[L]

            // ── Step A — Expert-cache planning & slot pinning ──────
            let plannedFetch = try runner.model.planRoutedExpertsIfPossible(
                layer: L, experts: experts)
            let routedOffsets = runner.model.routedExpertOffsets(layer: L)

            var pinnedSlots: [Int] = []
            if let plan = plannedFetch {
                pinnedSlots = plan.cachePlan.assignedSlots.filter { $0 >= 0 }
                runner.model.pinRoutedExpertSlots(layer: L, slots: pinnedSlots)
            }

            // ── Step B — Shared expert FFN → h1Buf ────────────────
            let sharedCB = queue.makeCommandBuffer()!
            runner.shared.encode(commandBuffer: sharedCB,
                                 x: denseX, xOffset: 0,
                                 gate: sharedProj.gate,
                                 up: sharedProj.up,
                                 down: sharedProj.down,
                                 y: runner.h1Buf, yOffset: 0,
                                 scratchGate: runner.denseScratchGate,
                                 scratchGateOffset: 0,
                                 scratchUp: runner.denseScratchUp,
                                 scratchUpOffset: 0,
                                 scratchAct: runner.denseScratchAct,
                                 scratchActOffset: 0)
            sharedCB.commit()

            // ── Step D — Async disk fetch for misses ──────────────
            // NOTE: We deliberately START this fetch BEFORE waiting for
            // sharedCB. fetchRoutedExperts kicks off disk I/O on a global
            // DispatchQueue; because of swift-async-algorithms-style
            // cooperative scheduling, the fetch runs concurrently with the
            // gate-compute path below, achieving the same I/O-compute overlap
            // that produceToken relies on.
            let blobs: [TensorView]
            if let plan = plannedFetch {
                blobs = try await runner.model.fetchRoutedExperts(plan: plan)
            } else {
                blobs = try await runner.model.fetchRoutedExperts(
                    layer: L, experts: experts)
            }

            // ── Step C — Apply sigmoid gate scalar (Qwen3.6 only) ─
            // Waits AFTER starting the async fetch, so shared FFN GPU work
            // overlaps with disk I/O above.
            if isQwen36 {
                if var gateWeights = runner.sharedExpertGate {
                    let w = try gateWeights.weight(layer: L)
                    // Persist back so subsequent layers reuse the cache
                    runner.sharedExpertGate = gateWeights

                    let dx = denseX.contents().assumingMemoryBound(to: Float16.self)
                    let gate = SharedExpertGateWeights.gateValue(
                        weight: w, x: dx, count: Int(D))

                    // Wait for shared FFN to complete before CPU modifies h1Buf
                    waitForCommandBuffer(sharedCB)
                    try checkCmdError(sharedCB.error)

                    let h1 = runner.h1Buf.contents().assumingMemoryBound(to: Float16.self)
                    for i in 0..<Int(D) {
                        h1[i] = Float16(Float(h1[i]) * gate)
                    }
                }
            }
            let routedBufs = blobs.map { $0.buffer }

            // ── Step E — Phase 1: load + activation into moeActs ──
            let routedCB = queue.makeCommandBuffer()!
            runner.moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: runner.moe.makeReusedRoutedArgumentBuffer(
                    routedBlobs: routedBufs, topK: topK),
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: denseX,
                acts: runner.moeActs,
                d: D, f: FmoE, topK: topK,
                groupSize: UInt32(runner.model.routedExpertGroupSize))

            // ── Step F — Phase 2: reduce + down-project → h2Buf ──
            runner.moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: runner.moe.makeReusedRoutedArgumentBuffer(
                    routedBlobs: routedBufs, topK: topK),
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                acts: runner.moeActs,
                routingWeights: runner.outWeights,
                residual: runner.zeroResidual,
                y: runner.h2Buf,
                d: D, f: FmoE, topK: topK,
                groupSize: UInt32(runner.model.routedExpertGroupSize))

            // ── Step G — Tail: hidden + h1 + routedScale*h2 ───────
            let tailCB = queue.makeCommandBuffer()!
            runner.encodeDecodeLayerTail(tailCB,
                                         postF2: nil, postF: nil,
                                         layerScalar: 1.0)
            tailCB.commit()

            // ── Step H — Sync + error check + unpin ───────────────
            waitForCommandBuffer(routedCB)
            try checkCmdError(routedCB.error)
            waitForCommandBuffer(tailCB)
            try checkCmdError(tailCB.error)
            if let plan = plannedFetch {
                runner.model.unpinRoutedExpertSlots(layer: L, slots: pinnedSlots)
            }
        }
        _ = currentLayer; _ = waitIfNeeded
    }
}
