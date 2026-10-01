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
- [x] **18.** build_base
- [x] **19.** run_variant
- [x] **20.** probe_rules
- [x] **21.** rltldr_experiment

## Results

```
1. The base policy
   public-world Pass@1 0.84 | hard tasks by construction: 41 training, 26 on held-out targets
   base Pass@1 at 16 attempts on the hard tasks, live world: 0.000

2. The learning barrier (GRPO, no insights)
   positives over 25 steps: 0 | no-insight Pass@1 train 0.000, held-out 0.000
   Every group has the same reward, so the advantage is zero and the weights never move.

3. Sequential attempts with insights (base policy)
   solved 28/41 training tasks within 6 attempts; mean first success at attempt 3.0
   insight advantage 0.32 | too-hard 0.32 | too-easy 0.12 | share of successes that were conditioned 1.00
   rail from ['clay', 'wooden_pickaxe']: insights ['craft_rail', 'craft_torch']
   minecart from ['furnace', 'iron_ingot', 'planks']: insights ['craft_wooden_pickaxe', 'craft_torch', 'craft_rail', 'craft_torch']
   iron_pickaxe from ['furnace', 'wood']: insights ['craft_torch', 'craft_stone_pickaxe', 'gather_stone', 'craft_torch', 'craft_iron_pickaxe', 'craft_torch']

4. RLTL;DR (lambda = 0.03)
   positives 566 | no-insight Pass@1 train 0.732, held-out targets 0.538
   after training: solved 0.95 of training groups | too-hard 0.11 | insight reliance 0.147 nats per token
   The policy now solves tasks it could never solve, with nothing in context, including targets it never trained on.

5. Where the gain comes from
   lambda 0.0  : positives  514 | no-insight Pass@1 train 0.640, held-out 0.510
   lambda 0.03 : positives  566 | no-insight Pass@1 train 0.732, held-out 0.538
   lambda 0.5  : positives  397 | no-insight Pass@1 train 0.000, held-out 0.010
   SFTL;DR on 705 task-insight pairs, no rollouts: train 0.000, held-out 0.000
   At this scale the transfer rides on the policy gradient of insight-conditioned successes; the insight-SFT term
   changes little when small and is harmful when large, and SFTL;DR alone does not move the unaided policy.
   The paper's policy is a pretrained LLM whose weights already link a sentence about a rule to the behavior that
   follows it, which is what lets a loss on the sentence change the behavior; a from-scratch policy lacks that link.

6. Probe: the hidden rules in the trained policy, held-out targets, no insight in context
   anvil    : success 1.00 | hidden tools crafted before they are needed 1.00 | insight head says: craft_torch
   clock    : success 0.88 | hidden tools crafted before they are needed 1.00 | insight head says: craft_torch
   compass  : success 0.75 | hidden tools crafted before they are needed 1.00 | insight head says: craft_torch
   jukebox  : success 0.00 | hidden tools crafted before they are needed 0.00 | insight head says: craft_torch
   The behavior transfers to targets never trained on, and the insight head names the rule most insights were about;
   a target that needs both hidden tools and thirty actions from an empty inventory is still out of reach.
```
