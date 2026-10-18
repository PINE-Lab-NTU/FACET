# Components

| Module | Function |
| --- | --- |
| `pretraining` | Alternating joint flow and VQA optimization |
| `model`, `action_wrench` | State/history conditioning and next-step joint targets |
| `refinement/reward` | Judge-based completion, timing and wrench penalties |
| `refinement/critic` | Distributional action–wrench values and auxiliary supervision |
| `refinement/credit` | Short-horizon credit and regime budgets |
| `refinement/objective` | Contact-weighted conditional and unconditional flow learning |
| `policy` | Checkpoint loading, guidance, contact features and serving |
| `policy/hierarchy` | Coarse joint proposals and contact-triggered refinement scheduling |
| `facet-online-adaptation` | Frozen token encoding and bounded local actor–critic learning |
