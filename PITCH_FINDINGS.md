# Pitch Findings — Team Yantra (for the deck)
*Consolidated for the teammate building the pitch. Depth: `REPORT.md` (method+results), `NEGATIVE_RESULTS.md` (the ~22-technique frontier map).*

---

## The one-paragraph story
Training-free, deterministic cross-modal MRI retrieval: **register every brain to a neutral template by image content (Mutual Information), describe structure with SSC-12 (modality-invariant self-similarity), trim the mismatched/resected voxels, and assign**. We found a **planted geometric leak** in dataset3, measured its exact worth, refused it, and re-earned the score from anatomy. Every dead end is documented with a measured reason — we didn't stumble onto the answer, we eliminated everything else.

## Per-level "what works where" (the method map)
| Level | What works | Why |
|---|---|---|
| **d1** (common grid) | SSC directly → ~1.0 | already co-registered; content matches |
| **d2** (indep rigid+elastic) | **registration is the lever** (none 0.08 → rigid 0.78 → affine-cascade **0.87**) + SSC + light trim | voxel correspondence destroyed → must re-align |
| **d3** (preop→intraop + resection) | content-registration + SSC + **heavy trim** (drops resected voxels) | tissue removed; match the intact anatomy |
| **descriptor** | **SSC-12 > MIND-6** | neighbour↔neighbour self-similarity, modality-invariant |
| **assignment** (batch only) | Hungarian / Sinkhorn / col-norm (transductive boost) | exploits the eval bijection — disclose, not "anatomy" |
| **deploy trust** | **Bayesian calibration + abstention** | flags uncertain d3 cases (conf↔correct +0.68) |

## Numbers — LB-VALIDATED (real Kaggle) ✅
| Claim | LB number | Status |
|---|---|---|
| Full submission, leak d3 | **1.000** | ✅ on board — *not selected* (it's the leak) |
| Full submission, honest d3 (Hungarian) | **0.914 / 0.930** | ✅ `full_honest` / `full_honest_v2` |
| **d3 leak** (resize / co-location) | **1.000** | ✅ |
| **d3 co-location BROKEN, no re-reg** | **0.259** | ✅ *the leak is worth 0.74 — proves the 1.0 is the grid* |
| **d3 honest** (co-location destroyed + content-registration) | **0.741** | ✅ `d3_strong` — earned by anatomy |
| Teammate v31 (Sinkhorn, deformable d3) | 0.923 | ✅ |

## Numbers — ORACLE-ONLY → being validated on Kaggle now ⏳
| Claim | Oracle value | Validation |
|---|---|---|
| **Greedy (single-query, deployable) macro** | ~0.80 (d1 1.0 / d2 0.81 / d3 0.58) | **submitting a greedy full submission now** to confirm on real LB |
| Hungarian − greedy gap (transductive boost) | +0.1–0.4 | confirmed by the greedy submission vs the 0.91 Hungarian one |

## The three pitch assets
1. **The method** — SSC + content-registration + trim. Training-free, deterministic, auditable (FDA/CE-credible). Per-level map above.
2. **Integrity** — found the planted d3 leak, **measured it (1.0 → 0.26 when broken)**, refused it, re-earned 0.74 by content-registration. Provable, not asserted.
3. **Rigor (the frontier map)** — `NEGATIVE_RESULTS.md`: ~22 techniques (8 classifiers incl. TabPFN, fusion, learned encoder, Bayesian, matrix-methods/Markov, GA, DP/DTW, NGF/NMI/shape/power-spectrum/radiomics) **all tested, all documented with why-we-tried-it → result → reason.** The bottleneck is the *content signal*, not the model. + **Bayesian calibration/abstention** as the deploy-time trust layer.

## Honest caveats (say them first)
- **Hungarian/Sinkhorn assignment is transductive** (assumes the eval N↔N bijection) — it's a disclosed leaderboard optimization, **not** deployable single-query anatomy. The deployable number is **greedy** (~0.80, being LB-confirmed).
- **Public LB is 27% (40/40/20 pools)** → private will derate; the assignment boost shrinks on larger pools. Because every level is *earned*, the derate is graceful.
- **Oracle is a proxy** — synthetic d3 resection is harsher than real, so some oracle numbers (e.g. optimal trim) don't transfer; we cross-check large claims on the real LB.
