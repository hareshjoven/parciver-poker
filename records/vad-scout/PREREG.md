# vad-scout: PREREG (25 Nov 2026), a scout, not a campaign

**Question.** Do VAD-CFR / WOP-CFR components (DeepMind, arXiv 2602.16928) or HS-DCFR schedules (arXiv 2404.09097)
reach lower exact exploitability than the live DCFR (alpha 1.5, beta 0, gamma 2, alternating updates) at the real-time
search budget of 100 and 400 iterations, on river and turn subgames?

**Prior: low.** On 29 Sep the turn solver already lost or tied against DCFR+, PCFR+ and other alpha/beta settings, and
gamma's gain levelled off at 5-8. This scout exists to find out if the newest schedules change that, cheaply.

**Setup.** `scripts/vad_scout.py`, a `GpuSolver` subclass (`src/` untouched), float32 on the GPU, exact exploitability in
chips per dealt pair. Spots: 6 river (`gpu_search.spec()`) and 6 turn (`small_turn_spec()`), boards seeded 7, 11, 23,
42, 77, 101, uniform and peaky (u^3) ranges alternating. Budgets T = 100 and 400 iterations, each a separate run
(schedules and the warm start depend on T).

**Arms.** `live`; the WOP components one at a time (`vneg` = sign-dependent negative discount t^-0.1/(t^-0.1+1),
`pow15` = policy power 1.5, `opt` = optimistic term 1/(1+t/100) times last instantaneous regret, `boost` = x1.1 on
positive instantaneous regret, `warm25` = hard warm start of 0.25 T, rescaled from the paper's 500 because 500
exceeds the whole budget); `wop` (vneg+pow15+opt+warm25); `vad` (wop+boost); `vad_nowarm`; `hs30`, `hs15`
(alpha = 1+3t/T, beta = -1-2t/T, gamma = 30-5t/T or 15-5t/T). The v2 volatility extras (EWMA alpha/beta/gamma, regret
floor) are not tested: the paper's v3 calls them secondary.

Interpretation choices (the paper text is ambiguous): the policy term d*R is read as the already-discounted R; the
optimistic term uses the stored (boosted) last instantaneous regret of the node's own player.

**Gate before any number is read.** River spots: the `live` arm must equal `GpuSolver` bit for bit (regret and strategy sum
after 50 iterations). Turn spots: `GpuSolver` itself is nondeterministic (`index_add_` atomics; two stock runs differ in
regrets but agree in exploitability to about 0.6%), so the gate there is exploitability within 3% of a stock run.
*Amended 25 Nov after the first launch aborted on the turn gate; no result had been read.* Turn results carry
about 0.5% run-to-run noise, small against the 10% bar.

**Pass rule (decides only whether a real pre-registration is worth writing).** An arm is "promising" if, at both T = 100
and T = 400, its exploitability is at most 0.90 x live on at least 9 of the 12 spots AND not above 1.00 x live on any
turn spot. Anything else is "no". A promising arm gets a proper PREREG with fresh spots and a power calculation before
anything touches the bot. Nothing here changes the live bot, and a pass would not be a novelty claim.

**Out of scope.** No bot change, no public release, no claim of a new method.
