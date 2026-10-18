# Policy server

Install the canonical FACET package and this client package in the policy environment.

```bash
python -m facet_adaptation.integrations.serve --backend facet0 --checkpoint /path/to/pi_ref --emit-prefix
facet-encoder extract --run runs/BASE_RUN --output /path/to/features --port 8000
facet-encoder train --embeddings /path/to/features --output /path/to/encoder --device cuda
python -m facet_adaptation.integrations.serve --backend facet0 --checkpoint /path/to/pi_ref --token-checkpoint /path/to/encoder/rl_token_step5000.pt
```

Features are fused state/wrench token outputs under the untagged instruction. Encoder
checkpoints bind to the exact policy identity and feature schema. The robot client uses
1=open and the model adapter converts the internal gripper convention once.
