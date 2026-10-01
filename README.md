# RLTL;DR in a Crafting World: Teach Yourself the Tech Tree

Apple's 2026 paper 'RLTL;DR: Self-improvement by Internalizing Self-generated Feedback', rebuilt in pure PyTorch on a crafting world. A tiny transformer policy knows the public tech tree, but the live world has hidden rules: mining iron ore now needs a torch, diamonds need a bucket. On those tasks the policy fails every attempt, GRPO gets no learning signal, and training stays flat. Build the paper's answer piece by piece: sequential attempts in which the verifier's diagnosis becomes a one-line insight, insight conditioning gated by the running success rate, the group-relative loss with positive-ratio filtering, and the flipped backpropagation mask that trains the task-to-insight mapping into the weights. Then run the experiments: the learning barrier, the breakthrough, Pass@1 without any insight in context on held-out targets, the lambda ablation, insight-only training, and a final probe that reads the hidden rules back out of the trained model.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** step_world
- [x] **2.** plan
- [x] **3.** hard_by_construction
- [x] **4.** insight_for
- [x] **5.** encode_context
- [x] **6.** Policy
- [x] **7.** sample_actions
- [x] **8.** pass_counts
- [x] **9.** demo
- [x] **10.** pretrain
- [x] **11.** sequential_groups
- [x] **12.** group_batch
- [x] **13.** positive_ratio_filter
- [x] **14.** rltldr_loss
- [x] **15.** train_rl
- [x] **16.** insight_metrics
- [x] **17.** sftldr

---

Built on Deep-ML.
