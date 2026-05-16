from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, FrozenSet

def apriori(
    transactions: List[List[str]],
    min_support: float,
) -> Dict[int, Dict[FrozenSet[str], float]]:
    n_tx = len(transactions)
    min_support_count = min_support * n_tx

    item_counts = defaultdict(int)
    for tx in transactions:
        for item in tx:
            item_counts[frozenset([item])] += 1

    def _filter_freq(item_count_map):
        return {
            itemset: count / n_tx
            for itemset, count in item_count_map.items()
            if count >= min_support_count
        }

    L = dict()
    L[1] = _filter_freq(item_counts)
    k = 2

    while L.get(k - 1):
        prev_fsets = list(L[k - 1].keys())
        candidates = set()
        for i in range(len(prev_fsets)):
            for j in range(i + 1, len(prev_fsets)):
                union = prev_fsets[i] | prev_fsets[j]
                if len(union) == k:
                    candidates.add(union)

        pruned_candidates = set()
        for c in candidates:
            all_subsets_frequent = all(
                (c - frozenset([item])) in L[k - 1]
                for item in c
            )
            if all_subsets_frequent:
                pruned_candidates.add(c)

        cand_counts = defaultdict(int)
        for tx in transactions:
            tx_fset = frozenset(tx)
            for cand in pruned_candidates:
                if cand.issubset(tx_fset):
                    cand_counts[cand] += 1

        Lk = _filter_freq(cand_counts)
        if Lk:
            L[k] = Lk
            k += 1
        else:
            break

    return L

def generate_rules(
    L: Dict[int, Dict[FrozenSet[str], float]],
    min_confidence: float,
) -> List[Tuple[FrozenSet[str], FrozenSet[str], float, float]]:
    rules = []
    support_lookup = {
        itemset: sup for level in L.values() for itemset, sup in level.items()
    }

    for k, itemsets in L.items():
        if k < 2:
            continue

        for itemset, sup_itemset in itemsets.items():
            for r in range(1, k):
                for antecedent in combinations(itemset, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    conf = sup_itemset / support_lookup[antecedent]
                    if conf >= min_confidence:
                        lift = conf / support_lookup[consequent]
                        rules.append((antecedent, consequent, conf, lift))

    rules.sort(key=lambda x: (-x[2], x[3]))
    return rules

def print_frequent_itemsets(L):
    print("Frequent itemsets:")
    for k in sorted(L):
        for itemset, sup in L[k].items():
            items = ", ".join(itemset)
            print(f"  [{items}]  (k={k}, support={sup:.3f})")
    print()

def print_rules(rules):
    print("Association rules:")
    for antecedent, consequent, conf, lift in rules:
        ant = ", ".join(antecedent)
        con = ", ".join(consequent)
        print(f"  [{ant}] → [{con}]  confidence={conf:.3f}, lift={lift:.3f}")
    print()

if __name__ == "__main__":
    transactions = [
        ["milk", "bread", "eggs"],
        ["beer", "bread"],
        ["milk", "diapers", "beer", "bread"],
        ["milk", "diapers", "beer", "cola"],
        ["bread", "milk", "diapers", "beer"],
        ["bread", "milk", "diapers", "cola"],
    ]

    min_support = 0.5
    min_confidence = 0.7

    L = apriori(transactions, min_support)
    print_frequent_itemsets(L)

    rules = generate_rules(L, min_confidence)
    print_rules(rules)