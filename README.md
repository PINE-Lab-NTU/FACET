<div align="center">

# Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2609.01596-b31b1b.svg)](https://arxiv.org/abs/2609.01596)
[![Project Page](https://img.shields.io/badge/Project-Page-4c8bf5.svg)](https://pine-lab-ntu.github.io/facet-0/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Facet--0-yellow.svg)](https://huggingface.co/Pinelab/Facet-0)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ManuFacet--1K-yellow.svg)](https://huggingface.co/datasets/Pinelab/ManuFacet-1K)

**[PINE Lab](https://pine-lab-ntu.github.io/), Nanyang Technological University, Singapore**

[Haoyuan Deng](https://denghaoyuan123.github.io/)\*,
[Haichao Liu](https://henryhcliu.github.io/)\*,
[Wenkai Guo](https://wkguo.github.io/)\*,
Yuan Ling,
[Zaijia Yang](https://yangzaijia.github.io/),
[Yuanjiang Xue](https://xuanyuan-jiang.github.io/),
Haosheng Sun,
Liangzi Wang,
[Ziwei Wang](https://ziweiwangthu.github.io/)†

<sub>\*Equal contribution &nbsp;·&nbsp; †Corresponding author</sub>

</div>

---

## 🚧 Code coming soon

---

## Overview

Real-world robotic assembly at sub-millimeter tolerances demands spatial precision, compliant interaction, and robustness to contact failures. **Facet-0** is a robotic foundation model that *predicts and values the contact consequences of its actions*.

It unifies multimodal representation learning and RL post-training around a **joint action–wrench proposal**: a causal wrench history is aligned with vision–language semantics and kinematic state, and flow matching generates each action chunk together with the future wrist-wrench profile it is expected to induce.

Force is not merely an input. It is an **observation**, an **objective**, and an **interface for actuation** — which is what lets the policy feel a jam and act on it.

## Results

On a five-task sub-millimeter computer-assembly benchmark (RAM, CPU, Disk, GPU, retention-bar close):

| Metric | Facet-0 | Baseline |
|---|---|---|
| Mean success (5 tasks) | **82%** | 15% (strongest), 10% (π₀.₅), 4% (GR00T N1.7) |
| Placement accuracy | **0.5 mm** | ≈5 mm (π₀.₅) |
| Command latency | **50 ms** | 150 ms (π₀.₅) |
| Few-shot transfer to an unseen module | **45%** | 5% (strongest) |

Controlled variants progress **16% → 38% → 82%** as semantic–contact alignment, value-guided refinement, and local adaptation are added in turn.

Against matched advantage-weighted behavior cloning on the same deployment corpus, critic-guided post-training lowers **intervention 47% → 24%** and raises **recovery 44% → 81%**.

Adaptation to an unseen memory module takes **ten demonstrations and about three hours**, updating **6.6% of the parameters**.

## ManuFacet-1K

A **1,000-hour** corpus of force-synchronized demonstrations and closed-loop rollouts, collected across **three robot embodiments**, **two chassis families**, and **multiple manufacturing cells** under one sensing and annotation protocol.

Every frame carries a 13-D proprioceptive state — end-effector pose, gripper opening, and six-axis wrench — so the corpus is force-annotated by construction. Segments are labeled with a skill-phase taxonomy, and failures, human takeovers, and recoveries are retained rather than filtered away.

🤗 [huggingface.co/datasets/Pinelab/ManuFacet-1K](https://huggingface.co/datasets/Pinelab/ManuFacet-1K)

## Citation

```bibtex
@article{deng2026facet0,
  title   = {Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation},
  author  = {Deng, Haoyuan and Liu, Haichao and Guo, Wenkai and Ling, Yuan and
             Yang, Zaijia and Xue, Yuanjiang and Sun, Haosheng and Wang, Liangzi and
             Wang, Ziwei},
  journal = {arXiv preprint arXiv:2609.01596},
  year    = {2026}
}
```

## Contact

Questions and collaboration enquiries: [ziwei.wang@ntu.edu.sg](mailto:ziwei.wang@ntu.edu.sg)
