"""
RLTL;DR in a Crafting World: Teach Yourself the Tech Tree

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - step_world
from collections import Counter

RAW = {
    "wood": None,
    "sand": None,
    "clay": None,
    "coal": "wooden_pickaxe",
    "stone": "wooden_pickaxe",
    "iron_ore": "stone_pickaxe",
    "redstone": "iron_pickaxe",
    "diamond": "iron_pickaxe",
}

RECIPES = {
    "planks": (["wood"], None),
    "chest": (["wood", "planks"], None),
    "wooden_pickaxe": (["planks", "wood"], None),
    "stone_pickaxe": (["stone", "planks"], None),
    "furnace": (["stone", "clay"], None),
    "torch": (["coal", "planks"], None),
    "glass": (["sand"], "furnace"),
    "brick": (["clay"], "furnace"),
    "iron_ingot": (["iron_ore"], "furnace"),
    "iron_pickaxe": (["iron_ingot", "planks"], None),
    "bucket": (["iron_ingot", "glass"], None),
    "rail": (["iron_ingot", "stone"], None),
    "anvil": (["iron_ingot", "brick"], None),
    "compass": (["iron_ingot", "redstone"], None),
    "minecart": (["iron_ingot", "rail"], None),
    "diamond_pickaxe": (["diamond", "planks"], None),
    "jukebox": (["chest", "diamond"], None),
    "clock": (["redstone", "glass"], None),
}

HIDDEN = {
    "gather_iron_ore": "torch",
    "gather_diamond": "bucket",
    "gather_redstone": "torch",
}


def step_world(inv, action, hidden=True):
    if action.startswith("gather_"):
        item = action[len("gather_"):]

        # Check the public gathering tool requirement first.
        required_tool = RAW[item]
        if required_tool is not None and inv[required_tool] <= 0:
            return False, required_tool

        # Check the hidden live-world requirement second.
        if hidden:
            hidden_tool = HIDDEN.get(action)
            if hidden_tool is not None and inv[hidden_tool] <= 0:
                return False, hidden_tool

        # Successful gathering adds one resource.
        inv[item] += 1
        return True, ""

    if action.startswith("craft_"):
        item = action[len("craft_"):]
        ingredients, required_tool = RECIPES[item]

        # Check the public crafting tool first.
        if required_tool is not None and inv[required_tool] <= 0:
            return False, required_tool

        # Check ingredients in the listed order.
        for ingredient in ingredients:
            if inv[ingredient] <= 0:
                return False, ingredient

        # All requirements are satisfied, so apply the craft.
        for ingredient in ingredients:
            inv[ingredient] -= 1

        inv[item] += 1
        return True, ""

    raise ValueError(f"Unknown action: {action}")


def simulate(have, actions, hidden=True):
    inv = Counter(have)
    log = {}

    for action in actions:
        if action == "END":
            break

        outcome = step_world(inv, action, hidden=hidden)
        log.setdefault(action, []).append(outcome)

    return inv, log


def succeeded(target, have, actions, hidden=True):
    final_inventory, _ = simulate(have, actions, hidden=hidden)
    return final_inventory[target] > 0

# Step 2 - plan
def plan(item, inv):
    # Nothing to do if the inventory already contains the item.
    if inv[item] > 0:
        return []

    actions = []

    # Raw resource: make the required public tool first, if necessary,
    # then gather the resource.
    if item in RAW:
        tool = RAW[item]

        if tool is not None and inv[tool] <= 0:
            actions.extend(plan(tool, inv))

        actions.append(f"gather_{item}")
        inv[item] += 1

        return actions

    # Craftable item: make the required tool first, if necessary.
    ingredients, tool = RECIPES[item]

    if tool is not None and inv[tool] <= 0:
        actions.extend(plan(tool, inv))

    # Resolve ingredients in recipe order.  After planning an ingredient,
    # consume exactly one unit because the eventual craft uses it.
    for ingredient in ingredients:
        if inv[ingredient] <= 0:
            actions.extend(plan(ingredient, inv))

        inv[ingredient] -= 1

    actions.append(f"craft_{item}")
    inv[item] += 1

    return actions


def plan_len(item, have=()):
    # Start from a fresh inventory initialized from `have`.
    inv = Counter(have)
    return len(plan(item, inv))


def sub_items(target):
    # Collect every recursive dependency: ingredients, tools, and
    # dependencies of those ingredients/tools.
    found = set()

    def visit(item):
        if item in found:
            return

        found.add(item)

        # Raw resource: its public tool is a dependency.
        if item in RAW:
            tool = RAW[item]
            if tool is not None:
                visit(tool)
            return

        # Craftable item: both its tool and ingredients are dependencies.
        ingredients, tool = RECIPES[item]

        if tool is not None:
            visit(tool)

        for ingredient in ingredients:
            visit(ingredient)

    visit(target)

    # The target itself is excluded from the dependency set.
    found.discard(target)

    return sorted(found)

# Step 3 - hard_by_construction
import random

def random_task(target, rng, max_have=3):
    subs = sub_items(target)

    k = rng.randint(0, min(max_have, len(subs)))
    have = tuple(sorted(rng.sample(subs, k)))

    return target, have


def hard_by_construction(tasks):
    hard = []

    for target, have in tasks:
        # Build the public expert plan from the given starting inventory.
        inv = Counter(have)
        actions = plan(target, inv)

        # The task is hard when that public-only plan fails in the
        # live world because of a hidden rule.
        if not succeeded(target, have, actions, hidden=True):
            hard.append((target, have))

    return hard


def split_tasks(hard, held_targets, rng):
    held_targets = set(held_targets)

    # Preserve the original order while partitioning the tasks.
    train = [task for task in hard if task[0] not in held_targets]
    eval_tasks = [task for task in hard if task[0] in held_targets]

    # Shuffle only the training portion.
    rng.shuffle(train)

    return train, eval_tasks

# Step 4 - insight_for
def insight_for(target, have, actions):
    # Simulate the full attempt in the live world.
    final_inv, log = simulate(have, actions, hidden=True)

    # No insight is required if the target was obtained.
    if final_inv[target] > 0:
        return None

    item = target
    last_act = None

    # Follow the target's dependency chain for at most 12 levels.
    for _ in range(12):
        if item in RECIPES:
            act = f"craft_{item}"
        else:
            act = f"gather_{item}"

        last_act = act
        attempts = log.get(act, [])

        # The action was never attempted.
        if not attempts:
            return act

        # If every attempt succeeded, the item was obtained but may
        # subsequently have been consumed.
        if all(ok for ok, _ in attempts):
            return act

        # Follow the missing item from the LAST FAILED attempt,
        # not merely the last attempt.
        failed = [(ok, missing) for ok, missing in attempts if not ok]
        _, missing = failed[-1]
        item = missing

    return last_act


def insight_text(action):
    action_type, item = action.split("_", 1)
    item = item.replace("_", " ")

    if action_type == "craft":
        return f"Obtain {item} first by crafting it."

    if action_type == "gather":
        return f"Obtain {item} first by gathering it."

    raise ValueError(f"Unknown action: {action}")

# Step 5 - encode_context
ITEMS = list(RAW) + list(RECIPES)

ACTIONS = (
    [f"gather_{r}" for r in RAW]
    + [f"craft_{c}" for c in RECIPES]
)

HINTS = [f"hint_{a}" for a in ACTIONS]

SPECIAL = ["PAD", "BOS", "TASK", "HAVE", "INS", "SEP", "END"]

VOCAB = SPECIAL + ITEMS + ACTIONS + HINTS

TOK = {name: idx for idx, name in enumerate(VOCAB)}

PAD, BOS, TASK, HAVE, INS, SEP, END = range(7)

ACTION_IDS = [TOK[action] for action in ACTIONS] + [END]


def encode_context(target, have, insights):
    ctx = [
        BOS,
        TASK,
        TOK[target],
        HAVE,
    ]

    # Include at most six starting items, in sorted order.
    for item in sorted(have)[:6]:
        ctx.append(TOK[item])

    # Encode each insight as INS followed by its dedicated hint token.
    for action in insights:
        ctx.extend([
            INS,
            TOK[f"hint_{action}"],
        ])

    ctx.append(SEP)

    return ctx


def sft_positions(ctx):
    # INS is the position whose next token is the corresponding hint.
    return [1 if tok == INS else 0 for tok in ctx]


def hint_token_mask(ctx):
    mask = [0] * len(ctx)

    # Every token immediately following INS is a hint token.
    for i in range(1, len(ctx)):
        if ctx[i - 1] == INS:
            mask[i] = 1

    return mask


def names(toks):
    return [VOCAB[tok] for tok in toks]

# Step 6 - Policy
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=2, max_len=72):
        super().__init__()

        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)

        self.max_len = max_len

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape

        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len={self.max_len}"
            )

        # Learned token and position embeddings.
        positions = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(positions).unsqueeze(0)

        # Upper-triangular boolean causal mask:
        # True entries block attention to future positions.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)

        return self.out(self.norm(h))

# Step 7 - sample_actions
ACTION_MASK = torch.full((len(VOCAB),), float("-inf"))
ACTION_MASK[ACTION_IDS] = 0.0


def action_logits(model, ctx_batch):
    # Pad all contexts to the same length.
    batch_size = len(ctx_batch)
    max_len = max(len(ctx) for ctx in ctx_batch)

    x = torch.full(
        (batch_size, max_len),
        PAD,
        dtype=torch.long,
    )

    lengths = []

    for i, ctx in enumerate(ctx_batch):
        x[i, :len(ctx)] = torch.tensor(ctx, dtype=torch.long)
        lengths.append(len(ctx))

    # Run the policy.
    logits = model(x)

    # Take the logits at each sequence's last real token.
    last_logits = torch.stack([
        logits[i, lengths[i] - 1]
        for i in range(batch_size)
    ])

    # Only ACTION_IDS (including END) may be sampled.
    return last_logits + ACTION_MASK.to(last_logits.device)


@torch.no_grad()
def sample_actions(model, contexts, max_steps, generator):
    # Work on copies so the caller's contexts are not modified.
    sequences = [list(ctx) for ctx in contexts]
    sampled = [[] for _ in contexts]
    finished = [False] * len(contexts)

    for _ in range(max_steps):
        active_indices = [
            i for i in range(len(sequences))
            if not finished[i]
        ]

        if not active_indices:
            break

        active_contexts = [sequences[i] for i in active_indices]
        logits = action_logits(model, active_contexts)

        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(
            probs,
            num_samples=1,
            generator=generator,
        ).squeeze(1)

        for row, idx in enumerate(active_indices):
            token = int(next_tokens[row].item())

            sequences[idx].append(token)
            sampled[idx].append(token)

            # Stop after END or when the sequence reaches
            # model.max_len - 1.
            if token == END or len(sequences[idx]) >= model.max_len - 1:
                finished[idx] = True

    return sampled

# Step 8 - pass_counts
@torch.no_grad()
def pass_counts(model, tasks, k, seed, max_steps=32, hidden=True):
    generator = torch.Generator().manual_seed(seed)

    contexts = []
    task_indices = []

    # Create k identical no-insight contexts for each task.
    for i, (target, have) in enumerate(tasks):
        ctx = encode_context(target, have, [])
        for _ in range(k):
            contexts.append(ctx)
            task_indices.append(i)

    # Sample all rollouts in one call.
    sampled = sample_actions(
        model,
        contexts,
        max_steps,
        generator,
    )

    counts = [0] * len(tasks)

    # Convert sampled token IDs back into action strings before
    # passing them to the crafting-world simulator.
    for token_ids, task_idx in zip(sampled, task_indices):
        actions = [VOCAB[token_id] for token_id in token_ids]

        target, have = tasks[task_idx]

        if succeeded(target, have, actions, hidden=hidden):
            counts[task_idx] += 1

    return counts


def pass_at_1(counts, k):
    if not counts:
        return 0.0

    return sum(count / k for count in counts) / len(counts)


def pass_at_k(counts):
    if not counts:
        return 0.0

    return sum(count > 0 for count in counts) / len(counts)

# Step 9 - demo
def demo(task, insights):
    target, have = task

    # Use one shared inventory for the complete demonstration.
    inv = Counter(have)
    actions = []

    # Obtain the item associated with each insight first, in order.
    for insight in insights:
        item = insight.split("_", 1)[1]
        actions.extend(plan(item, inv))

    # Finally obtain the actual target from the resulting inventory.
    actions.extend(plan(target, inv))
    actions.append("END")

    return actions


def make_batch(rows):
    # Find the longest token sequence.
    max_len = max(len(tokens) for tokens, _ in rows)

    batch_size = len(rows)
    seq_len = max_len - 1

    # X and Y are shifted by one position.
    X = torch.full(
        (batch_size, seq_len),
        PAD,
        dtype=torch.long,
    )
    Y = torch.full(
        (batch_size, seq_len),
        PAD,
        dtype=torch.long,
    )

    # M is aligned with Y and uses zero for padded positions.
    M = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.float32,
    )

    for i, (tokens, mask) in enumerate(rows):
        tokens = list(tokens)
        mask = list(mask)

        n = len(tokens) - 1

        X[i, :n] = torch.tensor(tokens[:-1], dtype=torch.long)
        Y[i, :n] = torch.tensor(tokens[1:], dtype=torch.long)
        M[i, :n] = torch.tensor(mask[1:], dtype=torch.float32)

    return X, Y, M

# Step 10 - pretrain
import math
import torch.nn.functional as F

def pretrain(model, rng, steps, batch=64, lr=3e-3):
    if steps <= 0:
        return 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    final_loss = 0.0

    for s in range(steps):
        # Set the cosine-scheduled learning rate for this step.
        step_lr = lr * 0.5 * (1.0 + math.cos(math.pi * s / steps))

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        rows = []

        for _ in range(batch):
            target = rng.choice(list(RECIPES))
            task = random_task(target, rng)

            # Draw the number of insights with the requested weighting.
            n = rng.choice([0, 0, 1, 1, 2, 3])
            insights = [rng.choice(ACTIONS) for _ in range(n)]

            ctx = encode_context(target, task[1], insights)
            actions = demo(task, insights)

            action_tokens = [TOK[action] for action in actions]
            tokens = ctx + action_tokens

            mask = hint_token_mask(ctx) + [1] * len(actions)

            rows.append((tokens, mask))

        X, Y, M = make_batch(rows)

        logits = model(X)

        # Compute token-level cross-entropy, then apply the training mask.
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            Y.reshape(-1),
            reduction="none",
        ).reshape_as(M)

        mask_total = M.sum()

        if mask_total.item() > 0:
            loss = (loss_per_token * M).sum() / mask_total
        else:
            loss = loss_per_token.sum() * 0.0

        optimizer.zero_grad()
        loss.backward()

        # Prevent excessively large parameter updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        final_loss = float(loss.item())

    return final_loss

# Step 11 - sequential_groups
def sequential_groups(
    model,
    tasks,
    K,
    generator,
    use_insights,
    max_steps=32,
    max_ins=6,
):
    # Chronological insight history for each task.
    insights = [[] for _ in tasks]

    # One group of rollout records per task.
    groups = [[] for _ in tasks]

    # Run K sequential attempts. Later attempts can depend on
    # successes and insights accumulated during earlier attempts.
    for k in range(K):
        contexts = []
        conditioned_flags = []
        context_insights = []

        for i, (target, have) in enumerate(tasks):
            ins = insights[i]
            succ = sum(
                1.0
                for rollout in groups[i]
                if rollout["reward"] > 0.0
            )

            # Insight conditioning is gated by the running success count.
            cond = (
                use_insights
                and len(ins) > 0
                and succ <= 0.5 * k
            )

            if cond:
                # Newest insight first, limited to max_ins entries.
                placed_ins = list(reversed(ins[-max_ins:]))
            else:
                placed_ins = []

            ctx = encode_context(target, have, placed_ins)

            contexts.append(ctx)
            conditioned_flags.append(cond)
            context_insights.append(placed_ins)

        # Sample one rollout for every task in this attempt.
        sampled = sample_actions(
            model,
            contexts,
            max_steps,
            generator,
        )

        for i, (target, have) in enumerate(tasks):
            acts = sampled[i]
            action_names = [VOCAB[token_id] for token_id in acts]

            reward = (
                1.0
                if succeeded(
                    target,
                    have,
                    action_names,
                    hidden=True,
                )
                else 0.0
            )

            record = {
                "ctx": contexts[i],
                "acts": acts,
                "reward": reward,
                "conditioned": conditioned_flags[i],
                "insights": context_insights[i],
            }

            groups[i].append(record)

            # On failure, let the verifier derive the next insight.
            if reward == 0.0 and use_insights:
                new_insight = insight_for(
                    target,
                    have,
                    action_names,
                )

                if (
                    new_insight is not None
                    and (
                        not insights[i]
                        or new_insight != insights[i][-1]
                    )
                ):
                    insights[i].append(new_insight)

    return groups, insights

# Step 12 - group_batch
def group_batch(groups):
    batch = []

    for group in groups:
        rewards = [rollout["reward"] for rollout in group]

        if not rewards:
            continue

        mean_reward = sum(rewards) / len(rewards)
        has_signal = any(
            reward != rewards[0]
            for reward in rewards[1:]
        )

        for rollout in group:
            if has_signal:
                rollout["adv"] = rollout["reward"] - mean_reward
            else:
                rollout["adv"] = 0.0

            # Keep every rollout from a signal-bearing group.
            # For zero-signal groups, keep only conditioned rollouts.
            if has_signal or rollout["conditioned"]:
                batch.append(rollout)

    return batch

# Step 13 - positive_ratio_filter
def positive_ratio_filter(rollouts, rng, ratio=0.75):
    pos = []
    neg = []
    zero = []

    for rollout in rollouts:
        adv = rollout["adv"]

        if adv > 0:
            pos.append(rollout)
        elif adv < 0:
            neg.append(rollout)
        else:
            zero.append(rollout)

    # Shuffle negatives before selecting the required prefix.
    rng.shuffle(neg)

    # Keep a fraction of negatives determined by the desired
    # positive-to-negative ratio.
    if pos:
        n_neg = int(len(pos) * (1.0 - ratio) / ratio)
    else:
        n_neg = 0

    kept_neg = neg[:n_neg]

    # Required order: positives, retained negatives, then zeros.
    return pos + kept_neg + zero

# Step 14 - rltldr_loss
@torch.no_grad()
def old_logprobs(model, rollouts):
    rows = []

    for rollout in rollouts:
        tokens = rollout["ctx"] + rollout["acts"]

        # The mask is irrelevant here; make_batch provides the
        # shifted X and Y tensors.
        mask = [0] * len(tokens)
        rows.append((tokens, mask))

    X, Y, _ = make_batch(rows)

    logits = model(X)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Log-probability of each target token.
    return torch.gather(
        log_probs,
        dim=-1,
        index=Y.unsqueeze(-1),
    ).squeeze(-1)


def rltldr_loss(model, rollouts, old_lp, lam, eps=0.2, use_grpo=True):
    rows = []

    for rollout in rollouts:
        ctx = rollout["ctx"]
        acts = rollout["acts"]
        tokens = ctx + acts

        # AM marks the predictions of sampled action tokens.
        am = [0] * len(ctx) + [1] * len(acts)
        rows.append((tokens, am))

    X, Y, AM = make_batch(rows)

    logits = model(X)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Current log-probabilities of the target tokens.
    lp = torch.gather(
        log_probs,
        dim=-1,
        index=Y.unsqueeze(-1),
    ).squeeze(-1)

    # Importance-sampling ratio.
    ratio = torch.exp(lp - old_lp)

    # Broadcast each rollout's scalar advantage across its tokens.
    A = torch.tensor(
        [rollout["adv"] for rollout in rollouts],
        dtype=lp.dtype,
        device=lp.device,
    ).unsqueeze(1)

    unclipped = ratio * A
    clipped = torch.clamp(
        ratio,
        1.0 - eps,
        1.0 + eps,
    ) * A

    # Negative clipped surrogate.
    surrogate = -torch.minimum(unclipped, clipped)

    # GRPO is evaluated only on action-token predictions.
    n_actions = AM.sum().clamp_min(1.0)
    grpo = (surrogate * AM).sum() / n_actions

    # Build the flipped SFT mask.  An INS at context position j
    # means the model predicts the hint at shifted target position j.
    SM = torch.zeros_like(AM)

    for i, rollout in enumerate(rollouts):
        ctx = rollout["ctx"]
        sft_mask = sft_positions(ctx)

        n = min(len(ctx) - 1, SM.shape[1])

        if n > 0:
            SM[i, :n] = torch.tensor(
                sft_mask[:n],
                dtype=SM.dtype,
                device=SM.device,
            )

    n_sft = int(SM.sum().item())

    # Negate after the masked mean so the zero-hint case remains -0.0,
    # matching the required reference behavior.
    n_sft_denom = SM.sum().clamp_min(1.0)
    sft = -((lp * SM).sum() / n_sft_denom)

    if use_grpo:
        loss = grpo + lam * sft
    else:
        loss = lam * sft

    return (
        loss,
        float(grpo.item()),
        float(sft.item()),
        n_sft,
    )

# Step 15 - train_rl
def train_rl(
    model,
    tasks,
    steps,
    cfg,
    seed,
    use_insights,
    lam,
    use_grpo=True,
    collect_pairs=None,
):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=0.0,
    )

    history = []

    for step in range(steps):
        # Sample at most tasks_per_step distinct tasks.
        n_tasks = min(cfg["tasks_per_step"], len(tasks))
        step_tasks = rng.sample(tasks, n_tasks) if n_tasks else []

        groups, insights = sequential_groups(
            model,
            step_tasks,
            cfg["K"],
            g,
            use_insights,
        )

        # Collect the insight lists actually used by conditioned rollouts.
        if collect_pairs is not None:
            for task_idx, group in enumerate(groups):
                for rollout in group:
                    if rollout["conditioned"]:
                        collect_pairs.append(
                            (
                                step_tasks[task_idx],
                                list(rollout["insights"]),
                            )
                        )

        # Count positive rollouts before any filtering.
        positives = sum(
            1
            for group in groups
            for rollout in group
            if rollout["reward"] > 0.0
        )

        # Form the group-relative batch and apply the positive-ratio filter.
        batch = group_batch(groups)
        batch = positive_ratio_filter(batch, rng)

        record = {
            "step": step,
            "positives": positives,
            "batch": len(batch),
        }

        if batch:
            # Compute old-policy log-probabilities once and reuse them
            # across all PPO epochs.
            old_lp = old_logprobs(model, batch)

            last_grpo = 0.0
            last_sft = 0.0
            last_sft_tokens = 0

            for _ in range(cfg["ppo_epochs"]):
                optimizer.zero_grad()

                loss, grpo, sft, sft_tokens = rltldr_loss(
                    model,
                    batch,
                    old_lp,
                    lam=lam,
                    use_grpo=use_grpo,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                last_grpo = grpo
                last_sft = sft
                last_sft_tokens = sft_tokens

            record["grpo"] = last_grpo
            record["sft"] = last_sft
            record["sft_tokens"] = last_sft_tokens

        history.append(record)

    return history

# Step 16 - insight_metrics
def insight_metrics(groups):
    n_groups = len(groups)

    # Overall fraction of groups with at least one successful rollout.
    solved_count = 0

    conditioned_groups = 0
    too_hard_count = 0
    too_easy_count = 0

    # Advantage is measured only where both conditioned and
    # unconditioned rollouts exist.
    advantage_values = []

    successful_rollouts = 0
    successful_conditioned = 0

    for group in groups:
        if any(rollout["reward"] > 0.0 for rollout in group):
            solved_count += 1

        conditioned = [
            rollout
            for rollout in group
            if rollout["conditioned"]
        ]
        unconditioned = [
            rollout
            for rollout in group
            if not rollout["conditioned"]
        ]

        if conditioned:
            conditioned_groups += 1

            conditioned_successes = sum(
                rollout["reward"] > 0.0
                for rollout in conditioned
            )

            conditioned_rate = (
                conditioned_successes / len(conditioned)
            )

            if conditioned_successes == 0:
                too_hard_count += 1

            if conditioned_successes == len(conditioned):
                too_easy_count += 1

            if unconditioned:
                unconditioned_successes = sum(
                    rollout["reward"] > 0.0
                    for rollout in unconditioned
                )

                unconditioned_rate = (
                    unconditioned_successes / len(unconditioned)
                )

                advantage_values.append(
                    conditioned_rate - unconditioned_rate
                )

        for rollout in group:
            if rollout["reward"] > 0.0:
                successful_rollouts += 1

                if rollout["conditioned"]:
                    successful_conditioned += 1

    solved = (
        solved_count / n_groups
        if n_groups
        else 0.0
    )

    too_hard = (
        too_hard_count / conditioned_groups
        if conditioned_groups
        else 0.0
    )

    too_easy = (
        too_easy_count / conditioned_groups
        if conditioned_groups
        else 0.0
    )

    advantage = (
        sum(advantage_values) / len(advantage_values)
        if advantage_values
        else 0.0
    )

    cond_share = (
        successful_conditioned / successful_rollouts
        if successful_rollouts
        else 0.0
    )

    return {
        "solved": solved,
        "too_hard": too_hard,
        "too_easy": too_easy,
        "advantage": advantage,
        "cond_share": cond_share,
    }


def strip_insights(ctx):
    stripped = []
    i = 0

    while i < len(ctx):
        # Remove INS and the hint token immediately following it.
        if ctx[i] == INS:
            i += 2
        else:
            stripped.append(ctx[i])
            i += 1

    return stripped


@torch.no_grad()
def insight_reliance(model, rollouts):
    rows_with_insights = []
    rows_without_insights = []
    action_lengths = []

    # Keep only successful conditioned rollouts.
    for rollout in rollouts:
        if not rollout["conditioned"]:
            continue

        if rollout["reward"] <= 0.0:
            continue

        acts = rollout["acts"]

        if not acts:
            continue

        ctx = rollout["ctx"]
        stripped = strip_insights(ctx)

        rows_with_insights.append(
            (ctx + acts, [0] * (len(ctx) + len(acts)))
        )
        rows_without_insights.append(
            (stripped + acts, [0] * (len(stripped) + len(acts)))
        )
        action_lengths.append(len(acts))

    if not rows_with_insights:
        return 0.0

    # Compute action log-probabilities with the original contexts.
    X_ctx, Y_ctx, _ = make_batch(rows_with_insights)
    logits_ctx = model(X_ctx)
    log_probs_ctx = torch.log_softmax(logits_ctx, dim=-1)
    token_lp_ctx = torch.gather(
        log_probs_ctx,
        dim=-1,
        index=Y_ctx.unsqueeze(-1),
    ).squeeze(-1)

    # Compute action log-probabilities after removing the insights.
    X_strip, Y_strip, _ = make_batch(rows_without_insights)
    logits_strip = model(X_strip)
    log_probs_strip = torch.log_softmax(logits_strip, dim=-1)
    token_lp_strip = torch.gather(
        log_probs_strip,
        dim=-1,
        index=Y_strip.unsqueeze(-1),
    ).squeeze(-1)

    gaps = []

    for i, (rollout, action_count) in enumerate(
        zip(
            [r for r in rollouts
             if r["conditioned"] and r["reward"] > 0.0 and r["acts"]],
            action_lengths,
        )
    ):
        ctx_len = len(rollout["ctx"])
        stripped_len = len(strip_insights(rollout["ctx"]))

        # In the shifted target sequence, the first action prediction
        # is at position len(context) - 1.
        start_ctx = ctx_len - 1
        end_ctx = start_ctx + action_count

        start_strip = stripped_len - 1
        end_strip = start_strip + action_count

        sum_ctx = token_lp_ctx[i, start_ctx:end_ctx].sum()
        sum_strip = token_lp_strip[i, start_strip:end_strip].sum()

        gap = (sum_ctx - sum_strip) / action_count
        gaps.append(float(gap.item()))

    return sum(gaps) / len(gaps) if gaps else 0.0

# Step 17 - sftldr
def sftldr(model, pairs, steps, lr, seed, batch=32):
    rng = random.Random(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    final_loss = 0.0

    if steps <= 0 or not pairs:
        return final_loss

    for _ in range(steps):
        rows = []

        # Sample pairs independently with replacement.
        for _ in range(batch):
            task, insights = rng.choice(pairs)
            target, have = task

            ctx = encode_context(
                target,
                have,
                insights,
            )

            rows.append(
                (
                    ctx,
                    hint_token_mask(ctx),
                )
            )

        X, Y, M = make_batch(rows)

        logits = model(X)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Log-probability of each target token.
        lp = torch.gather(
            log_probs,
            dim=-1,
            index=Y.unsqueeze(-1),
        ).squeeze(-1)

        # Minimize the masked mean negative log-probability
        # over hint-token targets only.
        mask_total = M.sum().clamp_min(1.0)
        loss = (-lp * M).sum() / mask_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())

    return final_loss

# Step 18 - build_base
def build_base(
    seed,
    pretrain_steps=600,
    per_target=10,
    held_targets=("compass", "jukebox", "clock", "anvil"),
):
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = Policy(len(VOCAB))

    # Pretrain the base policy using the seeded RNG.
    pretrain(
        model,
        rng,
        pretrain_steps,
    )

    # Generate random tasks in RECIPES insertion order.
    tasks = []
    seen = set()

    for target in RECIPES:
        for _ in range(per_target):
            task = random_task(target, rng)

            # Keep only the first occurrence of each distinct task.
            if task not in seen:
                seen.add(task)
                tasks.append(task)

    # Retain tasks for which the public expert fails in the live world.
    hard = hard_by_construction(tasks)

    # Held-out targets become evaluation tasks; the rest are shuffled
    # into the training set.
    train, evals = split_tasks(
        hard,
        set(held_targets),
        rng,
    )

    return model, train, evals

# Step 19 - run_variant
def run_variant(
    model,
    base_state,
    train,
    evals,
    steps,
    cfg,
    seed,
    use_insights,
    lam,
    use_grpo=True,
    collect_pairs=None,
    k=4,
    eval_seed=7,
):
    # Restore the common pretrained starting point.
    model.load_state_dict(base_state)

    # Train the requested RL variant.
    hist = train_rl(
        model,
        train,
        steps,
        cfg,
        seed=seed,
        use_insights=use_insights,
        lam=lam,
        use_grpo=use_grpo,
        collect_pairs=collect_pairs,
    )

    # Measure performance without providing insights at evaluation time.
    train_counts = pass_counts(
        model,
        train,
        k,
        seed=eval_seed,
        hidden=True,
    )

    eval_counts = pass_counts(
        model,
        evals,
        k,
        seed=eval_seed,
        hidden=True,
    )

    return {
        "positives": sum(h["positives"] for h in hist),
        "train_pass1": pass_at_1(train_counts, k),
        "eval_pass1": pass_at_1(eval_counts, k),
        "hist": hist,
    }

# Step 20 - probe_rules
def tool_before(acts, action):
    # The hidden rule for `action` specifies the tool that must be
    # crafted before the hidden-gather action can succeed.
    if action not in acts:
        return False

    hidden_tool = HIDDEN.get(action)

    if hidden_tool is None:
        return False

    required_craft = f"craft_{hidden_tool}"

    action_pos = acts.index(action)

    # The required craft action must occur before the first occurrence
    # of the hidden action.
    return required_craft in acts[:action_pos]


def hidden_actions(target):
    subs = set(sub_items(target))

    # Preserve the insertion order of HIDDEN.
    return [
        action
        for action in HIDDEN
        if action[len("gather_"):] in subs
    ]


@torch.no_grad()
def probe_rules(model, targets, seed, k=8, max_steps=32):
    generator = torch.Generator().manual_seed(seed)
    results = {}

    # Hint-token IDs, paired with their underlying action names.
    hint_pairs = [
        (TOK[f"hint_{action}"], action)
        for action in ACTIONS
    ]

    for target in targets:
        # Sample k no-insight rollouts from an empty inventory.
        ctx = encode_context(target, (), [])
        contexts = [ctx] * k

        sampled = sample_actions(
            model,
            contexts,
            max_steps,
            generator,
        )

        hidden = hidden_actions(target)

        successes = 0
        rule_first_count = 0

        for token_ids in sampled:
            acts = [VOCAB[token_id] for token_id in token_ids]

            if succeeded(
                target,
                (),
                acts,
                hidden=True,
            ):
                successes += 1

            if all(
                tool_before(acts, action)
                for action in hidden
            ):
                rule_first_count += 1

        # Probe the model at an explicit insight slot.
        probe_ctx = [
            BOS,
            TASK,
            TOK[target],
            HAVE,
            INS,
        ]

        x = torch.tensor(
            [probe_ctx],
            dtype=torch.long,
        )

        logits = model(x)[0, -1]

        hint_ids = [hint_id for hint_id, _ in hint_pairs]
        hint_logits = logits[hint_ids]

        top_idx = int(torch.argmax(hint_logits).item())
        top_hint = hint_pairs[top_idx][1]

        results[target] = {
            "success": successes / k if k else 0.0,
            "rule_first": (
                rule_first_count / k
                if k
                else 0.0
            ),
            "top_hint": top_hint,
        }

    return results

