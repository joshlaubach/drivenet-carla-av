# Workflow: Causal Analysis via Propensity Score Matching

## Objective
Estimate the causal effect of driving conditions (rain, night lighting, urban
terrain, PPO vs BC training) on model performance using propensity score
matching. Report average treatment effects (ATEs) with bootstrap confidence
intervals and Rosenbaum sensitivity bounds.

## Required Inputs
- `results/eval_results.json` populated by the EvaluationAgent

## Tools
| Tool | Call | Purpose |
|---|---|---|
| `json.load` | Load eval_results.json | Input data |
| `sklearn.linear_model.LogisticRegression` | Fit propensity model | Estimate P(treatment \| covariates) |
| `scipy.spatial.cKDTree` | 1-nearest-neighbour matching | Match treated/control units |
| `numpy` bootstrap | Resample matched pairs | Bootstrap CIs |
| `json.dump` | Save causal_results.json | Output |

## Treatments Under Study
Treatment definitions are loaded from `configs/causal.yaml`. The agent
reconstructs filter functions from the declarative DSL at runtime.

| Treatment | Treated condition | Control condition | Outcome |
|---|---|---|---|
| Rain | weather = HardRainNoon | weather = ClearNoon | route_completion |
| Night | weather = ClearNight | weather = ClearNoon | route_completion |
| Urban | town in {Town01, Town03} | town = Town05 | collision_rate |
| PPO | model_type = ppo | model_type = bc | route_completion |
| Driving Style | model_type = ppo AND driving_style = chill | model_type = bc | route_completion |

Note: Traffic density is collected as a condition variable in data collection
(NB01) but is not varied in the evaluation grid (NB04), so it cannot serve as
a treatment or meaningful covariate in this analysis.

## Covariates (confounders) for matching
- model_type (BC/PPO as 0/1 for non-PPO treatments)
- town (label-encoded)
- weather (label-encoded, for non-weather treatments)
- driving_style (label-encoded, available when eval records include the field)

## Expected Outputs
```
results/
    causal_results.json     # ATE, 95% CI, Rosenbaum Gamma per treatment
    causal_plots/
        psm_rain.png
        psm_night.png
        psm_urban.png
        psm_ppo.png
        psm_driving_style.png
```

Each result entry:
```json
{
  "treatment": "rain",
  "n_treated": 48,
  "n_control_matched": 48,
  "ate": -0.12,
  "ci_lower": -0.18,
  "ci_upper": -0.06,
  "rosenbaum_gamma": 1.4
}
```

## Sequencing
1. Load `eval_results.json`; parse into a DataFrame.
2. For each treatment:
   a. Define treatment indicator T and covariate matrix X.
   b. Fit LogisticRegression on (X, T); compute propensity scores P(T=1|X).
   c. Match each treated unit to its nearest control unit by propensity score
      (1:1 without replacement via cKDTree).
   d. Compute ATE = mean(outcome_treated) - mean(outcome_matched_control).
   e. Bootstrap ATE 10,000 times (resample matched pairs with replacement);
      record 2.5th and 97.5th percentiles as 95% CI.
   f. Compute Rosenbaum Gamma: find smallest Gamma such that the Wilcoxon signed-rank
      test on matched pairs exceeds p=0.05 under worst-case unmeasured confounding.
3. Save results to `causal_results.json`.
4. Plot propensity score distributions (before/after matching) for each treatment.

## Edge Cases
- **Insufficient treated units** (< 20): skip treatment and log warning.
- **Propensity scores all near 0 or 1**: matching quality is poor; log overlap
  diagnostic and interpret ATE with caution.
- **Perfect separation in logistic regression**: use C=1.0 regularization to
  prevent degenerate scores.

## Caveats
- **Q3 (Urban)**: Only 2 of the 3 evaluation towns (Town01, Town03) are
  classified as urban. The strong correlation between town identity and the
  urban treatment may limit the power of confounder adjustment.
- **Q4 (PPO vs BC)**: BC and PPO models were trained under different condition
  distributions (BC on all 324 conditions; PPO on a curriculum starting from
  ClearNoon). PSM controls for town and weather at evaluation time, but cannot
  adjust for training-distribution differences. Interpret the PPO vs BC ATE
  as conditional on the evaluation grid, not as a general training-method effect.
- **Q5 (Driving Style)**: The driving_style treatment compares comfort-optimized
  PPO (chill) against BC. Since chill deprioritizes speed and penalizes aggressive
  maneuvers, any route completion deficit may reflect the intentional trade-off
  between comfort and efficiency rather than a model capability gap.
