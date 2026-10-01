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

