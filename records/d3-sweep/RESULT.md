# D3-SWEEP result

**Pin: d3s-rw.** d3s-rw was confirmed at seed 1.

Control (N1, `d2b-w2048-wd0.1`): loss 0.011799, real 0.002217, random 0.021381. Win bar: loss <= 0.011563 and real <= 0.002217.

| arm | loss | vs N1 | real | random | MAE % pot | wins |
|---|---|---|---|---|---|---|
| `d3s-drop05` | 0.012114 | +2.67% | 0.002565 | 0.021663 | 4.298 | no |
| `d3s-drop10` | 0.012310 | +4.34% | 0.003016 | 0.021605 | 4.416 | no |
| `d3s-ema999` | 0.011801 | +0.01% | 0.002217 | 0.021384 | 4.160 | no |
| `d3s-ema9999` | 0.011771 | -0.24% | 0.002236 | 0.021306 | 4.143 | no |
| `d3s-rw` | 0.011320 | -4.06% | 0.002181 | 0.020460 | 3.598 | yes |

Best of each winning family: {'loss': 'd3s-rw'}. Best single winner: d3s-rw. Combo: not called.
Confirmation (`d3s-confirm`, seed 1, recipe {"dropout": 0.0, "ema": 0.0, "loss": "rw"}): loss 0.011468 (bar 0.011817), real 0.002196 (control 0.002213): CONFIRMED.

Limits (accepted in the registration): winner's curse on the shared selection half; a weak confirmation (one seed pair); the real-row clause is close to a coin flip for a small effect; the range-weighted arm is trained toward MAE and judged on the plain Huber, and stands in for D3's "EV-space loss" (it is not DEVN). A recipe tuned at 951k rows is carried, not proven, at ~1.9M: Gate A on D1c's gate half tests it.

Next: an amendment to records/d3/PREREG.md naming only this pin, committed before D1c's first label is read; `N2_ARM` in scripts/d3.py updated to match.
