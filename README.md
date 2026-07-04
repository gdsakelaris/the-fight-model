# A Leak-Safe, Time-Aware Ensemble for UFC Fight Outcome and Method Prediction

A machine-learning system that predicts **(1) the winner** of a UFC bout and
**(2) the method of victory** (Decision / KO-TKO / Submission), built and
evaluated on strictly chronological data with no information leakage.

The full implementation is a single, self-contained script:
[`UFC_ML_Model.py`](UFC_ML_Model.py).

## Headline results

From the most recent full run (dataset through July 2026; 9,197 fights, of
which 7,873 rows from the selected 2010+ training era):

| Metric (500 most-recent-fight holdout) | Value |
|---|---|
| Winner accuracy | **69.0%** |
| Winner log-loss | 0.6102 |
| Brier score | 0.2107 |
| Expected calibration error (ECE) | 0.0644 |
| Method accuracy (given winner pick correct) | **52.2%** (majority baseline: 41.7%) |
| Method macro-F1 (given winner pick correct) | 49.0% |

Walk-forward diagnostics (six expanding-window folds retraining a single
gradient-boosted model per fold) show accuracy improving monotonically toward
the present — 59.5% on the earliest fold to 67.2% on the most recent —
confirming that skill is not an artifact of one lucky holdout window.

These numbers are reproduced by `python UFC_ML_Model.py --train-only` (see
[Reproducing results](#reproducing-results)).

## How it works

### 1. Leak-safe chronological feature engineering

Every fight row is constructed exclusively from each fighter's **pre-fight**
state; running state is updated only *after* a row is emitted, so no feature
can contain information from its own fight. Features include:

- **Career and recency statistics** — striking/grappling rates, Bayesian-shrunk
  accuracies, round-by-round pacing and cardio profiles, finish-round
  distributions, durability and submission composites, and
  opponent-baseline-adjusted (`oba_*`) outputs that credit production against
  each opponent's own pre-fight statistical norms.
- **Glicko-2 ratings**, optionally updated with a continuous
  **margin-of-victory score** (finish timing + box-score dominance) so dominant
  wins move ratings more than split decisions — without ever flipping the true
  result.
- **Elo and divisional Elo**, with divisional rank snapshots and rating-trajectory
  slopes.
- **Phase Glicko ratings** — four ratings per fighter (striking-offense,
  striking-defense, grappling-offense, grappling-defense) updated bipartite
  within every fight, so "elite wrestler with weak takedown defense" is
  representable and matchups cross offense against the opponent's defense.
- **Style-matchup records** — each fighter's historical win rate against
  opponents bucketed by style (striker / wrestler / submission hunter).
- **Altitude and acclimatization** — venue elevation combined with each
  fighter's training-camp elevation (shock of fighting above camp, benefit of
  descending from a high camp).
- **Context finish priors** — time-decayed finish rates for fights of the same
  weight class / gender / rounds / title status.

A **feature-routing table** (`FEATURE_ROUTING`) declares which of the three
models (winner, finish-vs-decision, KO-vs-sub) is allowed to see each
engineered feature, so outcome-conditional method signals never add noise to
the winner model and vice versa.

### 2. Corner-swap-symmetric winner ensemble

UFC data has a strong red-corner convention (the promoted fighter is red). The
winner model is made exactly **corner-symmetric**: training data is augmented
with every fight seen from both corners (antisymmetric features negated,
probability features complemented, raw red/blue pairs exchanged), and at
inference each prediction is the average of the forward and corner-swapped
passes. Base models — LightGBM, XGBoost, CatBoost (default + Optuna-tuned
variants), HistGradientBoosting, RandomForest, and ExtraTrees — are combined
by a simplex-weighted blend fit on time-series out-of-fold predictions, with
Platt/isotonic calibration fit on early validation data and accepted only if
it helps on late validation data.

Feature selection is two-stage and leak-safe (fit on training rows only):
correlation pruning (|r| > 0.95, keeping the more target-relevant member of
each pair) followed by Meinshausen–Bühlmann stability selection over 15
bootstrap runs.

### 3. Two-stage method model

Method prediction is conditioned on the predicted winner: the feature matrix
is re-oriented so the picked winner occupies the positive corner, then a
**Stage 1** classifier (Finish vs Decision) and **Stage 2** classifier (KO vs
Submission, trained on finishes only) — each a subsample-bagged
HistGradientBoosting ensemble with RandomForest/ExtraTrees side models — are
combined with direct multiclass, one-vs-rest, and linear heads plus history-,
group-, and submission-signal priors. The ~20 blend hyperparameters are tuned
by Optuna against a chunked walk-forward objective, and a
**champion/challenger gate** persists the best configuration across runs so
retraining noise cannot silently degrade the deployed blend.

### 4. Evaluation protocol

- Chronological split by **fixed fight counts**: the 500 most-recent fights are
  an untouched holdout; the 600 before them are the validation window; the
  rest is training. Fixed counts keep the evaluation windows identical across
  training-era candidates, making era selection a like-for-like comparison.
- **Strict future mode** (default ON) forbids every holdout-informed choice:
  the combiner, calibrator, and decision threshold are selected on validation
  data only.
- Trained stages are **cached content-addressed** on the SHA-256 of the
  dataset contents plus every configuration value that affects the result, so
  a cache hit is only possible when retraining would reproduce the identical
  model.

## Repository contents

| Path | Description |
|---|---|
| `UFC_ML_Model.py` | The complete model: features, ratings, training, evaluation, GUI. |
| `pure_fight_data_with_event_and_camp_altitudes.csv` | Fight dataset (see [Data](#data)). |
| `Stat Definitions.txt` | Data dictionary: a definition for every dataset column. |
| `results/` | Audit artifacts and a sample prediction export from the published run. |
| `requirements.txt` | Pinned Python dependencies. |
| `CITATION.cff` | Citation metadata. |
| `.ufc_model_cache/` | Generated: cached training stages (safe to delete; forces retrain). |
| `UFC_Predictions.xlsx` | Generated: prediction export from the GUI. |

## Requirements

- Python ≥ 3.10 (developed on 3.13); Tkinter (bundled with standard CPython
  installers) is needed only for the GUI.
- `pip install -r requirements.txt`

LightGBM, XGBoost, CatBoost, and Optuna are technically optional — the script
degrades gracefully if any is missing — but the published results use all of
them.

## Data

One row per fight (~9,200 UFC fights, 1993–present), with:

- event metadata: date, location, venue elevation, weight class, gender,
  title-bout flag, scheduled rounds;
- outcome: winner corner, method string, finish round, fight time;
- per-corner box score: significant/total strikes (landed + attempted),
  knockdowns, takedowns, submission attempts, reversals, control time, and
  strike distribution by target (head/body/leg) and position
  (distance/clinch/ground) — as fight totals and per round (`r_rd1_*`, …);
- fighter attributes: height, reach, weight, stance, age at event, and
  training-camp elevation.

Every column is defined in [`Stat Definitions.txt`](Stat%20Definitions.txt).

Statistics were collected from [ufcstats.com](http://ufcstats.com) (the UFC's
official statistics provider) and enriched with venue and training-camp
elevations via geocoding. The CSV in this repository is the exact file the
published results were computed from; its SHA-256 participates in the cache
keys, so any modification forces a retrain.

## Reproducing results

```bash
pip install -r requirements.txt
python UFC_ML_Model.py --train-only
```

The script prints a structured report: split contract, feature pruning,
Optuna tuning, combiner selection, calibration, holdout evaluation, method
evaluation (conditioned on winner pick), and walk-forward diagnostics.

Notes:

- **Determinism.** All RNGs are seeded (`RANDOM_SEED = 42`); repeated runs on
  the same data and configuration reproduce the same results and hit the
  stage caches. Minor cross-platform variation is possible from multithreaded
  tree libraries.
- **Runtime.** A cold run performs the full Optuna search (80 trials × 3
  libraries for the winner stage, 80 × 2 + 400 blend trials for the method
  stage) — expect several hours on a desktop CPU. Subsequent runs replay the
  cached stages in minutes.
- **GUI.** `python UFC_ML_Model.py` (no flags) opens the prediction GUI: one
  matchup per line as `Red,Blue,Weight Class,Gender,Rounds[,Elevation ft or
  Location]`, with optional American moneyline odds after each name
  (auto-detected). Predictions are exported to `UFC_Predictions.xlsx`.

### Configuration switches (controlled A/B tests)

Documented ablations are exposed as environment variables; defaults reproduce
the published configuration.

| Variable | Default | Effect |
|---|---|---|
| `UFC_MOV_ENABLED` | `1` | Margin-of-victory-adjusted ratings (`0` = hard win/loss only). |
| `UFC_MOV_MODE` | `full` | MOV recipe: `full` (finish timing + box score) or `buckets` (low-overfit baseline). |
| `UFC_PHASE_RATINGS` | `1` | Four-dimensional phase Glicko ratings on/off. |
| `UFC_OBA_ENABLED` | `1` | Opponent-baseline-adjusted (`oba_*`) features on/off. |
| `UFC_CORNER_CORRECTION` | `0` | One-parameter red-corner logit correction (tested; defaults off). |
| `UFC_WINNER_ALL_FEATURES` | `0` | Winner model bypasses feature routing (tested; defaults off). |
| `UFC_COMBINER_ROBUST` | `1` | Restrict combiner to simplex blends (`0` re-allows stacker meta-learners). |

Key in-file constants: `FORCED_START_YEAR` (training-era pin; `None`
re-enables automatic era selection), `VAL_FIGHTS` / `TEST_FIGHTS` (split
sizes), `OPTUNA_TRIALS`, `METHOD_TUNING_TRIALS`, and `STRICT_KEEP_MODELS`
(the active base-model lineup). Each is documented where it is defined.

## Limitations

- Fighter identity is name-based; distinct fighters sharing a name would be
  merged.
- Debutants have no history: the model falls back to prior features, and the
  GUI skips matchups involving debutants by default.
- The method model's rarest class (Submission) remains the hardest
  (F1 ≈ 0.33 on the holdout); results are reported per class.
- Betting-value utilities are provided for research completeness; nothing in
  this repository constitutes betting advice.

## Reproducibility statement

All results in the accompanying paper were produced by this repository at the
pinned dependency versions in `requirements.txt`, from the included dataset,
with all random seeds fixed. Code was developed by the author with assistance
from Anthropic's Claude (AI pair programming); all modeling decisions were
validated empirically through the ablations and audits described above.

## Citation

If you use this code or dataset, please cite it (see `CITATION.cff`).

## License

The code is released under the [MIT License](LICENSE).

The dataset consists of factual fight statistics compiled from
[ufcstats.com](http://ufcstats.com), enriched with geocoded elevations, and is
included solely to make the published results reproducible. UFC® is a
trademark of Zuffa, LLC; this project is not affiliated with or endorsed by
the UFC.
