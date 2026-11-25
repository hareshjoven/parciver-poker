# D3 scouts S0a and S0c: results (10 Nov 2026, run after the recipe pin, before any D1c compute)

Both ran on commit a14559d with the GPU free of other jobs (S0a's labelling 18:48-19:19 SGT; its scoring afterwards, `gpu_peers_after` empty). Records: `s0a.json`, `s0c.json`.
Rules were fixed in PREREG 5.1 and the arms before any query existed; nothing here is a new decision.

## S0a (what the live flop search asks the net) -> **MIXED branch**
- 200 spots (seed block 780000; 216 recorded, the first 200 used), 18 cells, live flop search with N1 leaves ("sym", gamma 5, 500 iterations, `graph=False`), 10 queries per
  spot from iteration 500's two evaluate calls: 2,000 queries, labelled at D1b's settings (500 iterations, gamma 5, `--terminals64`; 1.06 solves/s, mean exploitability 0.17% pot).
- N1's selection loss on them **0.053663 = 24.21 x 0.002217; 95% CI of r [21.71, 26.85]** (bootstrap over spots, `default_rng(0)`, 10,000). The line is 1.5: lower end 21.71 > 1.5.
- **D1c = half recorded, half harvested:** 18 x 13,890 = 250,020 recorded situations with twins = 500,040 rows, plus 499,968 harvested rows (`turn_data.py generate --samples 1000008`, seed 700007 / 800007 as PREREG 5.1).
- Diagnostic (read-only, not a rule): the gap is broad, not outliers. Median per-query loss 0.0356 (16x the real-range figure), p90 0.110, p99 0.324; the top 10% of queries carry 37% of the loss;
  per-spot means run 0.0098-0.204 (median 0.0396); every one of the 18 cells is above 0.025 (range 0.025-0.099). N1 is worse on live queries than on D1b's random twins (0.0214).
- Caveat: v2-8B could not play either stake in the recorder (`ValueError`), so every spot used v2-2B (the PREREG fallback); S0a cannot see an 8B difference.

## S0c (ensemble of N1 and d2b-seed1, old selection half, 23,728 rows, forward only)
- N1 0.011799, d2b-seed1 0.011937, mean prediction 0.011116: **gain 5.79% vs N1 (and vs the better single)**; under the 6% line, so **distillation is not proposed**.
