"""
RLTL;DR in a Crafting World: Teach Yourself the Tech Tree scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""RLTL;DR (Kirchhof et al., Apple, 2026) in a crafting world.

Story: a tiny transformer policy knows the public tech tree but not the live
world's hidden rules (iron ore and redstone need a torch, diamonds need a bucket).
On the tasks that need a hidden rule every attempt fails, so GRPO has no signal.
Sequential attempts with the verifier's one-line insight in context break through,
the policy gradient on those successes is internalized, and the trained policy
solves held-out targets with no insight in context. The lambda sweep and the
SFTL;DR control locate where the gain comes from at this scale, and a probe reads
the hidden rules back out of the trained policy.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    cfg = dict(lr=5e-4, tasks_per_step=8, K=6, ppo_epochs=2)
    res = rltldr_experiment(0, pretrain_steps=600, rl_steps=25, cfg=cfg)

    print("1. The base policy")
    print(f"   public-world Pass@1 {res['public_pass1']:.2f} | hard tasks by construction: {res['n_train']} training, {res['n_eval']} on held-out targets")
    print(f"   base Pass@1 at 16 attempts on the hard tasks, live world: {res['base_pass16']:.3f}")

    b = res["barrier"]
    print("\n2. The learning barrier (GRPO, no insights)")
    print(f"   positives over {len(b['hist'])} steps: {b['positives']} | no-insight Pass@1 train {b['train_pass1']:.3f}, held-out {b['eval_pass1']:.3f}")
    print("   Every group has the same reward, so the advantage is zero and the weights never move.")

    e = res["explore"]
    m = e["metrics"]
    print("\n3. Sequential attempts with insights (base policy)")
    print(f"   solved {e['solved']}/{res['n_train']} training tasks within {cfg['K']} attempts; mean first success at attempt {e['first']:.1f}")
    print(f"   insight advantage {m['advantage']:.2f} | too-hard {m['too_hard']:.2f} | too-easy {m['too_easy']:.2f} | share of successes that were conditioned {m['cond_share']:.2f}")
    for task, ins in e["examples"]:
        print(f"   {task[0]} from {list(task[1]) or 'nothing'}: insights {ins}")

    v = res["variants"]
    full = v[0.03]
    print("\n4. RLTL;DR (lambda = 0.03)")
    print(f"   positives {full['positives']} | no-insight Pass@1 train {full['train_pass1']:.3f}, held-out targets {full['eval_pass1']:.3f}")
    t = res["trained_metrics"]
    print(f"   after training: solved {t['solved']:.2f} of training groups | too-hard {t['too_hard']:.2f} | insight reliance {t['reliance']:.3f} nats per token")
    print("   The policy now solves tasks it could never solve, with nothing in context, including targets it never trained on.")

    print("\n5. Where the gain comes from")
    for lam in sorted(v):
        r = v[lam]
        print(f"   lambda {lam:<5}: positives {r['positives']:4d} | no-insight Pass@1 train {r['train_pass1']:.3f}, held-out {r['eval_pass1']:.3f}")
    s = res["sftldr_offline"]
    print(f"   SFTL;DR on {s['pairs']} task-insight pairs, no rollouts: train {s['train_pass1']:.3f}, held-out {s['eval_pass1']:.3f}")
    print("   At this scale the transfer rides on the policy gradient of insight-conditioned successes; the insight-SFT term")
    print("   changes little when small and is harmful when large, and SFTL;DR alone does not move the unaided policy.")
    print("   The paper's policy is a pretrained LLM whose weights already link a sentence about a rule to the behavior that")
    print("   follows it, which is what lets a loss on the sentence change the behavior; a from-scratch policy lacks that link.")

    print("\n6. Probe: the hidden rules in the trained policy, held-out targets, no insight in context")
    for target, p in res["probe"].items():
        print(f"   {target:9s}: success {p['success']:.2f} | hidden tools crafted before they are needed {p['rule_first']:.2f} | insight head says: {p['top_hint']}")
    print("   The behavior transfers to targets never trained on, and the insight head names the rule most insights were about;")
    print("   a target that needs both hidden tools and thirty actions from an empty inventory is still out of reach.")


if __name__ == "__main__":
    main()

