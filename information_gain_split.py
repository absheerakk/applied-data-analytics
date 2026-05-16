information_gain_split.py
"""
Find the best split point for a numeric attribute using Information Gain
-----------------------------------------------------------------------
Given:
    • values : 1D list/array of numeric feature values
    • labels : 1D list/array of class labels (any hashable type)

Returns:
    best_threshold, best_IG
Author: <you>
"""

import math
from collections import Counter
from typing import Sequence, Tuple, Any, List

def entropy(class_counts: Counter) -> float:
    """Shannon entropy from a Counter of class counts."""
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in class_counts.values() if c)

def best_numeric_split(values: Sequence[float],
                       labels: Sequence[Any],
                       min_leaf: int = 1
                       ) -> Tuple[float, float]:
    """
    Evaluate every midpoint between sorted values (where the class label changes)
    and return the threshold that maximizes Information Gain.
    """
    # 1. Sort by feature value (keep labels in sync)
    sorted_pairs = sorted(zip(values, labels), key=lambda x: x[0])
    sorted_vals, sorted_labs = zip(*sorted_pairs)

    # 2. Parent entropy
    parent_counts = Counter(sorted_labs)
    parent_ent = entropy(parent_counts)

    best_gain = -float("inf")
    best_thresh = None

    # 3. Running class counts
    left_counts = Counter()
    right_counts = parent_counts.copy()
    n = len(values)

    # 4. Scan possible split positions
    for i in range(n - 1):
        val, lab = sorted_vals[i], sorted_labs[i]
        left_counts[lab]  += 1
        right_counts[lab] -= 1

        left_size  = i + 1
        right_size = n - left_size

        # Skip invalid splits
        if left_size < min_leaf or right_size < min_leaf:
            continue
        if sorted_vals[i] == sorted_vals[i + 1]:
            continue

        # Midpoint threshold
        threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0

        # Child entropies
        ent_left  = entropy(left_counts)
        ent_right = entropy(right_counts)

        # Weighted average
        weighted_child_ent = ((left_size / n) * ent_left +
                              (right_size / n) * ent_right)

        # Information Gain
        gain = parent_ent - weighted_child_ent

        if gain > best_gain:
            best_gain   = gain
            best_thresh = threshold

    return best_thresh, best_gain


# ----------------------------------------------------------------------
# 🔎 Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Toy data: numeric feature (temperature) and binary class (play tennis)
    temperatures: List[float] = [85, 80, 83, 70, 68, 65, 72, 69,
                                 75, 75, 72, 81, 71, 71]
    play: List[str] = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                       "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    split, ig = best_numeric_split(temperatures, play, min_leaf=2)

    print(f"Best split point : {split:.2f}")
    print(f"Information Gain : {ig:.4f}")
