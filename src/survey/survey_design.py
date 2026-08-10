from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable


# A 3-edge-colouring of the Fano-plane incidence graph.
# Each triple contains method indices in display order A, B, C.
# Across all seven blocks:
# - each method occurs in exactly three blocks;
# - every method pair occurs together exactly once;
# - every method appears exactly once as A, once as B and once as C.
ORDERED_FANO_BLOCKS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 0, 4),
    (5, 6, 0),
    (1, 5, 3),
    (6, 4, 1),
    (2, 3, 6),
    (4, 2, 5),
)


def validate_design() -> None:
    method_counts = Counter(i for block in ORDERED_FANO_BLOCKS for i in block)
    pair_counts = Counter(
        pair
        for block in ORDERED_FANO_BLOCKS
        for pair in combinations(sorted(block), 2)
    )
    position_counts = {
        method: Counter(
            position
            for block in ORDERED_FANO_BLOCKS
            for position, value in enumerate(block)
            if value == method
        )
        for method in range(7)
    }

    assert method_counts == Counter({i: 3 for i in range(7)})
    assert len(pair_counts) == 21 and set(pair_counts.values()) == {1}
    assert all(counts == Counter({0: 1, 1: 1, 2: 1}) for counts in position_counts.values())


def build_variant_rows(
    argument_ids: Iterable[str],
    method_order: list[str],
) -> list[dict]:
    """Create seven balanced questionnaire variants.

    Every variant presents seven arguments and all seven Fano blocks once.
    Argument order is cyclically shifted between variants. For each fixed
    argument, the seven variants assign all seven blocks exactly once.
    """
    validate_design()
    argument_ids = [str(value) for value in argument_ids]
    if len(argument_ids) != 7:
        raise ValueError("The exact design requires exactly seven arguments.")
    if len(method_order) != 7 or len(set(method_order)) != 7:
        raise ValueError("Exactly seven unique methods are required.")

    rows: list[dict] = []
    for display_position in range(7):
        for version_zero_based in range(7):
            argument_index = (display_position + version_zero_based) % 7
            block_index = (argument_index + version_zero_based) % 7
            ordered_method_indices = ORDERED_FANO_BLOCKS[block_index]
            ordered_methods = [method_order[i] for i in ordered_method_indices]

            rows.append(
                {
                    "version": version_zero_based + 1,
                    "display_position": display_position + 1,
                    "argument_index": argument_index + 1,
                    "global_row_id": argument_ids[argument_index],
                    "block_index": block_index + 1,
                    "method_A": ordered_methods[0],
                    "method_B": ordered_methods[1],
                    "method_C": ordered_methods[2],
                }
            )
    return rows
