# GPU solver kernels: what landed and what did not

The real-time search solves river, turn and flop subgames with a fused GPU implementation of discounted CFR (DCFR) on a single consumer desktop (RTX 4070, Ryzen 5800X). This document covers the kernel-level work done to make that solver faster or more precise, under pre-registered gates. Outcome: two changes landed, a want-only-sums change and Q4, which computes the showdown and fold terminal sums with float-float (double-float32) pairs and made a live turn solve 2.89x faster than the float64 version it replaced while matching its precision. The neural-net leaf work (N1, N1b) did not land, and several other candidates were parked. Code: `src/parciver/gpu_fused.py`, `src/parciver/flop_leaves.py`, `scripts/kernel_bench.py`.

## Setting and rules

- The kernels sit near the memory roofline (400-450 GB/s measured against about 504 GB/s peak), so more speed needs fewer memory words or fewer multiply-adds per iteration.
- The card runs float64 at 1/64 of its float32 rate and has 36 MB of L2 cache.
- Typical live turn jobs are 8.5k-11.7k rows by 1,128 hands with 100 iterations; flop spots have 57 nodes and 9 leaves, so a net batch of 441 per traversal.

Every candidate was held to the same rules, written down before measurement:

1. **Same bits or an oracle.** Either bit-identical to the current code on regret, strategy sum and values after 1, 2, 3 and 100 iterations (all options, and recorded real jobs reproducing their recorded answers), or tested against a float64 oracle.
2. **Precision gate.** The candidate's float32 error against the oracle must not exceed the current code's, in mean and in max over nodes. A secondary whole-solve check required the average strategy's exploitability to stay within 0.5% (relative) of the current code at 100 and 400 iterations.
3. **Speed gate.** An interleaved ABBA benchmark: 12 rounds, median ratio at least 1.02, and at least 10 of 12 rounds faster. Two amendments were added later: the GPU may be at most 60% busy during timing, and both arms must run byte-identical code except for the change under test.

## What landed

**Want-only sums** (compute only the strategy-sum slots the caller wants). Bit-identical to the previous code. Live solve 137.5 -> 119.1 ms (ratio 1.156, 12/12 rounds); per-iteration speed not slower (river 1.037, turn agent menus 1.005, turn full menus 1.000, flop 1.000).

**Q3: float64 terminal sums** (offline only, switch `terminals64=True`). The precision work targeted the terminal-node sums (showdown and fold values).
- Accumulating them in float64 cut the error: terminal mean 4.204e-7 -> 4.164e-8, max 1.415e-6 -> 5.954e-8.
- Holdings outside the error bound went from 820,994,938 to 0 of 1,410,996,024.
- Whole-solve exploitability barely moved: 11.288 -> 11.065% at 100 iterations, 1.4758 -> 1.4767% at 400.
- The price was speed: live solve 105.0 -> 409.1 ms (ratio 0.257), so Q3 stayed an offline tool. Under Q3 the terminals took 87.5% of an iteration (6.25 ms; `_showdown64` 4.29 ms, `_fold64` 1.18 ms).

**Q4: float-float pairs for the terminal sums** (live since 11 Oct). Each value is carried as an unevaluated sum of two float32s, giving near-float64 accuracy at float32 rates.
- First attempt, 8 Oct: the whole-solve test failed (+1.93% and +1.91% against a 0.5% bound). The registration was flawed: the per-start log-ratio standard deviation was 4.6% at 100 iterations and 7.2% at 400, so the bound sat at noise level. A real bug was also found (the compiler fused a multiply-add, which the CPU interpreter hid) and fixed (`fma(h,m,l*m)`, `enable_fp_fusion=False`).
- Re-registered gates S1-S5, 9 Oct, all passed:
  - S1 (speed, against Q3): live solve 408.38 -> 141.01 ms (2.89x, 12/12 rounds); river 1.74x (0.1198 -> 0.0687 ms); turn agent menus 2.84x (4.867 -> 1.715 ms); turn full menus 2.73x (11.68 -> 4.27 ms).
  - S3 (A/A control, same code twice): n=100 -0.005% [-0.35, +0.34]; n=400 -0.157% [-1.24, +0.93].
  - S4 (whole-solve exploitability, one-sided upper bound): n=100 +0.060% (upper +0.34%), n=400 -1.172% (upper -0.41%), both PASS against a margin of ln 1.02 = +1.98%.
  - S5: 25 of 25 rows within limits, the worst at 40% of its limit (Q3 was at 83%).

## What did not land

**N1 / N1b: faster value-net forward at flop leaves** (11 Oct, final net). The net forward pass is 84% of a 4.39-4.44 ms flop iteration.
- The factored forward ran 1.90-1.97x slower (median 1.95x; 8.40-8.66 ms).
- Precision (rule 2), mean / max error: dense 1.7280e-7 / 1.26464e-6; N1 2.4989e-7 / 1.39047e-6 (worse); N1b 8.3135e-8 / 1.26033e-6; N1c 8.2066e-8 / 1.26033e-6.
- Rule 3 did not complete: 40 factored tables of about 230 MB each filled the 12 GB card, Windows paged GPU memory, and the run was stopped after 1 h 44 min.
- Receipts: `records/kernels-n1-final/`.

Other attempts at the same bottleneck:
- cuBLAS SGEMM is already at its floor at M = 441.
- A Triton FP32 GEMM was 1.1x faster but 4.6x less accurate, so it was dropped.
- bfloat16 leaves were slower and moved leaf values by about 2 pots.
- A Triton scouting estimate of about 1.08x per iteration was never built; at least 1.22x was needed.

**Parked, not built or not run:** K10 (keeping river subtrees resident in the 36 MB L2 cache) passed its pre-gate (C = 1.154; T 2.153 ms, down+up 0.791, Q4 terminals 1.337 vs float32 0.681, g 0.767 us). It would save at most 13% of a replayed iteration, about 9% of a turn solve, and its probe was not run (`records/k10-pregate/RESULT.md`). Also parked: P4, K1-K3, N1c, N2, N3.

## Limits

- All timings are one machine and one GPU; ratios from interleaved ABBA runs are more trustworthy than absolute milliseconds.
- Q3's accuracy gain barely changed whole-solve exploitability, which is why precision was treated as a gate rather than a goal. Q4's value is speed at preserved accuracy.
- The generic float64 CUDA solver is not deterministic run to run (see `docs/verification.md`), so oracle comparisons use fixed inputs and bounds, not bit equality, for the non-identical candidates.
- The first Q4 registration is an example of a gate set at noise level (a 0.5% bound against per-start noise of 4.6-7.2%); the re-registration added an A/A control (S3) and a margin of ln 1.02.
