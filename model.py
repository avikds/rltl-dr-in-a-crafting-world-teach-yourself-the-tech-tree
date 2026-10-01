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
    # Execute the complete attempt in the live world.
    final_inv, log = simulate(have, actions, hidden=True)

    # No insight is needed when the target was successfully obtained.
    if final_inv[target] > 0:
        return None

    item = target
    last_act = None

    # Follow the dependency chain for at most twelve levels.
    for _ in range(12):
        if item in RECIPES:
            act = f"craft_{item}"
        else:
            act = f"gather_{item}"

        last_act = act
        attempts = log.get(act, [])

        # The action was never attempted, so it is the next thing
        # the agent should obtain.
        if not attempts:
            return act

        # If every recorded attempt succeeded, the item was produced
        # but may have been consumed later in the full attempt.
        if all(ok for ok, _ in attempts):
            return act

        # Follow the missing prerequisite from the last failed attempt.
        _, missing = attempts[-1]
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

