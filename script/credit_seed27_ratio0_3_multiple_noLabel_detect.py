#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
from collections import Counter, defaultdict

import pandas as pd

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Utilities
# -----------------------------
VALID_ERROR_TYPES = {
    "form-based",
    "polluted",
    "distorted",
    "synonymous",
    "collateral",
    "homonymous",
}


def _require_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Robust parsing; keep NaT for invalid timestamps
    return pd.to_datetime(series, errors="coerce")


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _lower_alnum_space(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _normalize_spaces(s)


def _add_error(df: pd.DataFrame, idx, err_type: str):
    if err_type not in VALID_ERROR_TYPES:
        raise ValueError(f"Invalid error type: {err_type}")
    df.at[idx, "error_detected"] = True
    current = df.at[idx, "detected_error_types"]
    if not current:
        df.at[idx, "detected_error_types"] = err_type
    else:
        parts = current.split("|")
        if err_type not in parts:
            parts.append(err_type)
            df.at[idx, "detected_error_types"] = "|".join(parts)


# -----------------------------
# Pollution detection + base extraction
# -----------------------------
# Heuristics: detect machine-generated tokens appended/prepended via separators or brackets,
# containing long digit sequences, timestamp-like fragments, random alphanumerics, or resource-like IDs.

SEP_CHARS = r"[_\-\.\|:/#]"
BRACKETED_TOKEN_RE = re.compile(r"(\[[^\]]+\]|\([^)]+\))")

# Timestamp-ish fragments: 20230929, 20230929T131145981, 20230929 130852312000, 20230930_001115044, 091144221
TS_FRAGMENT_RE = re.compile(
    r"(?:(?:19|20)\d{2}[01]\d[0-3]\d(?:[T _-]?\d{6,12})?)|(?:\d{6,12})"
)

# Long digit sequences (likely IDs)
LONG_DIGITS_RE = re.compile(r"\d{7,}")

# Random-ish alphanumerics (mixed letters+digits) length >= 6
MIXED_ALNUM_RE = re.compile(r"(?=.*[a-zA-Z])(?=.*\d)[a-zA-Z0-9]{6,}")

# Resource-like prefix: Manager000001_Approve request, Clerk-000001:Something
RESOURCE_PREFIX_RE = re.compile(r"^[A-Za-z]+-?\d{4,}[_\-\s:|/]+")

# Token separators
SPLIT_RE = re.compile(SEP_CHARS)


def _token_looks_machine_generated(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    # Remove surrounding brackets/parentheses for evaluation
    t2 = t.strip("[](){}<>")
    if not t2:
        return False

    # Strong signals
    if TS_FRAGMENT_RE.search(t2):
        return True
    if LONG_DIGITS_RE.search(t2):
        return True
    if MIXED_ALNUM_RE.fullmatch(t2) is not None:
        return True

    # Many digits relative to length
    digits = sum(ch.isdigit() for ch in t2)
    if len(t2) >= 6 and digits / max(1, len(t2)) >= 0.5:
        return True

    # Looks like resource id chunk
    if re.fullmatch(r"[A-Za-z]+-?\d{4,}", t2):
        return True

    return False


def extract_base_label_and_polluted(activity: str):
    """
    Returns: (base_label, is_polluted)
    base_label is a cleaned label with machine-generated prefix/suffix removed when confidently separable.
    """
    if activity is None or (isinstance(activity, float) and pd.isna(activity)):
        return "", False

    raw = str(activity)
    s = _normalize_spaces(raw)

    if not s:
        return "", False

    # Remove resource-like prefix if present
    s_no_prefix = s
    if RESOURCE_PREFIX_RE.search(s):
        s_no_prefix = RESOURCE_PREFIX_RE.sub("", s).strip()

    # If bracketed tokens exist and any look machine-generated, remove them
    bracketed = BRACKETED_TOKEN_RE.findall(s_no_prefix)
    removed_any_bracketed = False
    if bracketed:
        s_tmp = s_no_prefix
        for bt in bracketed:
            if _token_looks_machine_generated(bt):
                s_tmp = s_tmp.replace(bt, " ")
                removed_any_bracketed = True
        s_no_prefix = _normalize_spaces(s_tmp)

    # Split by separators to detect suffix/prefix tokens
    # Keep original words too; only remove tokens that look machine-generated.
    parts = [p for p in SPLIT_RE.split(s_no_prefix) if p is not None]
    parts = [_normalize_spaces(p) for p in parts]
    parts = [p for p in parts if p]

    # If no separators and no bracketed removal and no resource prefix removal, likely not polluted
    had_prefix_removed = (s_no_prefix != s) and bool(RESOURCE_PREFIX_RE.search(s))
    had_any_sep = bool(SPLIT_RE.search(s_no_prefix))
    if not had_any_sep and not removed_any_bracketed and not had_prefix_removed:
        return s_no_prefix, False

    # Identify machine tokens among parts
    machine_flags = [_token_looks_machine_generated(p) for p in parts]
    if not any(machine_flags) and not removed_any_bracketed and not had_prefix_removed:
        # separators alone are not enough
        return s_no_prefix, False

    # Remove machine tokens but preserve meaningful text
    kept = [p for p, is_m in zip(parts, machine_flags) if not is_m]

    # If we removed everything, fall back to original (avoid empty base)
    base = _normalize_spaces(" ".join(kept)) if kept else s_no_prefix

    # Additional cleanup: if base still contains obvious timestamp fragments at end, trim them
    base2 = base
    base2 = re.sub(rf"(?:\s+{TS_FRAGMENT_RE.pattern})+$", "", base2).strip()
    base2 = _normalize_spaces(base2)

    # Decide polluted if we removed something meaningful (machine token/bracket/prefix)
    is_polluted = (base2 != _normalize_spaces(s)) and (removed_any_bracketed or had_prefix_removed or any(machine_flags))
    return base2 if base2 else base, bool(is_polluted)


# -----------------------------
# Distortion detection (character-level)
# -----------------------------
def _levenshtein_distance(a: str, b: str) -> int:
    # Iterative DP, O(min(len(a),len(b))) space
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    # now len(a) >= len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _edit_similarity(a: str, b: str) -> float:
    a2 = a or ""
    b2 = b or ""
    if not a2 and not b2:
        return 1.0
    dist = _levenshtein_distance(a2, b2)
    denom = max(len(a2), len(b2), 1)
    return 1.0 - (dist / denom)


def detect_distortions(clean_labels_series: pd.Series, min_sim=0.86, max_len_diff_ratio=0.5):
    """
    Returns mapping: label -> canonical_label (for distorted labels only)
    Canonical candidates are chosen among cleaned labels by frequency and stability.
    """
    labels = clean_labels_series.fillna("").astype(str).map(_normalize_spaces)
    freq = Counter(labels)
    unique = [l for l in freq.keys() if l]

    # Precompute normalized forms for comparison (lower + collapse spaces)
    norm = {l: _normalize_spaces(l.lower()) for l in unique}

    # Build candidate canonical pool: all unique labels (do not assume small/frequent only)
    # For each label, find a "better" close neighbor with higher frequency and very high similarity.
    distorted_to_canon = {}

    # Sort by increasing frequency so we can map rarer to more frequent when close
    unique_sorted = sorted(unique, key=lambda x: (freq[x], len(x)))

    for l in unique_sorted:
        best = None
        best_score = -1.0
        nl = norm[l]
        for c in unique:
            if c == l:
                continue
            # Prefer more frequent and "stable" (shorter or equal length) as canonical
            if freq[c] < freq[l]:
                continue

            nc = norm[c]
            # Quick length filter
            la, lb = len(nl), len(nc)
            if max(la, lb) == 0:
                continue
            if abs(la - lb) / max(la, lb) > max_len_diff_ratio:
                continue

            sim = _edit_similarity(nl, nc)
            if sim >= min_sim:
                # Tie-break: higher freq, then higher sim, then shorter label
                score = (freq[c], sim, -len(c))
                if best is None or score > best_score:
                    best = c
                    best_score = score

        if best is not None:
            # Avoid mapping if labels are identical after normalization (case/space only)
            if nl != norm[best]:
                distorted_to_canon[l] = best

    return distorted_to_canon


# -----------------------------
# Synonym detection (semantic)
# -----------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def detect_synonyms(clean_labels_series: pd.Series, sim_threshold=0.80):
    """
    Returns:
      - noncanonical_to_canon: dict label -> canonical_label for labels in synonym clusters (excluding canonical)
      - clusters: list of sets (for potential debugging)
    """
    labels = clean_labels_series.fillna("").astype(str).map(_normalize_spaces)
    freq = Counter(labels)
    unique = [l for l in freq.keys() if l]
    if len(unique) < 2:
        return {}, []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    uf = UnionFind(range(len(unique)))

    # Build edges for pairs above threshold
    # O(n^2) - acceptable for typical activity vocab sizes; still guard for very large.
    n = len(unique)
    if n > 5000:
        # Safety: avoid quadratic blow-up; in such case, skip synonym detection rather than be incorrect.
        return {}, []

    for i in range(n):
        # Only j>i
        row = sim_matrix[i]
        for j in range(i + 1, n):
            if float(row[j]) >= sim_threshold:
                uf.union(i, j)

    comps = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)

    clusters = []
    noncanonical_to_canon = {}

    for comp_idxs in comps.values():
        if len(comp_idxs) < 2:
            continue
        cluster_labels = [unique[i] for i in comp_idxs]
        clusters.append(set(cluster_labels))

        # Canonical = most frequent; tie-break by shorter then lexicographic
        canonical = sorted(
            cluster_labels,
            key=lambda x: (-freq[x], len(x), x.lower()),
        )[0]

        for l in cluster_labels:
            if l != canonical:
                noncanonical_to_canon[l] = canonical

    return noncanonical_to_canon, clusters


# -----------------------------
# Form-based detection (timestamp collisions within case)
# -----------------------------
def detect_form_based(df: pd.DataFrame, case_col="Case", act_col="Activity", ts_col="Timestamp"):
    """
    Mark all events in (case, timestamp) groups where:
      - group size >= 2
      - at least 2 distinct activity labels in the group
    """
    # Use raw activity (not cleaned) because definition is about different activity labels recorded at same timestamp.
    # Still normalize spaces to avoid trivial differences.
    tmp = df[[case_col, act_col, ts_col]].copy()
    tmp["_act_norm"] = tmp[act_col].fillna("").astype(str).map(_normalize_spaces)
    tmp["_ts"] = _safe_to_datetime(tmp[ts_col])

    # Only consider valid timestamps
    valid = tmp["_ts"].notna()
    tmp = tmp[valid]

    grp = tmp.groupby([case_col, "_ts"], sort=False)
    idxs_to_flag = set()
    for (_, _), g in grp:
        if len(g) >= 2 and g["_act_norm"].nunique(dropna=False) >= 2:
            idxs_to_flag.update(g.index.tolist())
    return idxs_to_flag


# -----------------------------
# Collateral detection (bursts within case)
# -----------------------------
def detect_collateral(df: pd.DataFrame, case_col="Case", ts_col="Timestamp", res_col="Resource", window_seconds=5):
    """
    Within each case, sort by timestamp (stable). Find consecutive bursts where:
      - consecutive events within <= window_seconds OR identical timestamps
      - and share same resource (strong contextual similarity)
    Flag all events in bursts of size >= 2.
    """
    tmp = df[[case_col, ts_col, res_col]].copy()
    tmp["_ts"] = _safe_to_datetime(tmp[ts_col])
    tmp["_res"] = tmp[res_col].fillna("").astype(str).map(_normalize_spaces)

    idxs_to_flag = set()

    for case_id, g in tmp.groupby(case_col, sort=False):
        g2 = g.copy()
        # Keep original order for ties/NaT; sort by timestamp but stable
        g2["_orig_idx"] = g2.index
        # Put NaT at end; they cannot be used for time deltas reliably
        g2 = g2.sort_values(by=["_ts", "_orig_idx"], kind="mergesort")

        # Iterate consecutive
        burst = [g2.iloc[0]] if len(g2) > 0 else []
        for i in range(1, len(g2)):
            prev = g2.iloc[i - 1]
            cur = g2.iloc[i]

            prev_ts, cur_ts = prev["_ts"], cur["_ts"]
            prev_res, cur_res = prev["_res"], cur["_res"]

            close_in_time = False
            if pd.notna(prev_ts) and pd.notna(cur_ts):
                delta = (cur_ts - prev_ts).total_seconds()
                close_in_time = (delta <= window_seconds) or (delta == 0)
            # Require same resource as "strong contextual similarity"
            same_res = (prev_res != "" and prev_res == cur_res)

            if close_in_time and same_res:
                burst.append(cur)
            else:
                if len(burst) >= 2:
                    for row in burst:
                        idxs_to_flag.add(int(row["_orig_idx"]))
                burst = [cur]

        if len(burst) >= 2:
            for row in burst:
                idxs_to_flag.add(int(row["_orig_idx"]))

    return idxs_to_flag


# -----------------------------
# Homonymous detection (contextual path divergence within case)
# -----------------------------
def detect_homonymous(df: pd.DataFrame, case_col="Case", act_col="Activity", ts_col="Timestamp"):
    """
    For each case, sort by timestamp (stable). For labels that occur exactly twice in the case:
      - compare immediately preceding activity labels (normalized)
      - if preceding labels are both present and different, flag both occurrences as homonymous
    Prefer collateral when occurrences are 3+ (do not flag homonymous in that situation).
    """
    tmp = df[[case_col, act_col, ts_col]].copy()
    tmp["_ts"] = _safe_to_datetime(tmp[ts_col])
    tmp["_act_norm"] = tmp[act_col].fillna("").astype(str).map(_normalize_spaces)

    idxs_to_flag = set()

    for case_id, g in tmp.groupby(case_col, sort=False):
        g2 = g.copy()
        g2["_orig_idx"] = g2.index
        g2 = g2.sort_values(by=["_ts", "_orig_idx"], kind="mergesort")

        acts = g2["_act_norm"].tolist()
        orig_idxs = g2["_orig_idx"].tolist()

        positions_by_label = defaultdict(list)
        for pos, a in enumerate(acts):
            positions_by_label[a].append(pos)

        for label, poss in positions_by_label.items():
            if not label:
                continue
            if len(poss) != 2:
                # If 3+ occurrences, do not flag homonymous (prefer collateral per instruction)
                continue
            p1, p2 = poss
            if p1 == 0 or p2 == 0:
                continue  # no preceding activity for one occurrence
            prev1 = acts[p1 - 1]
            prev2 = acts[p2 - 1]
            if prev1 and prev2 and prev1 != prev2:
                idxs_to_flag.add(int(orig_idxs[p1]))
                idxs_to_flag.add(int(orig_idxs[p2]))

    return idxs_to_flag


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python script.py input.csv output.csv")

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv(in_path)

    _require_columns(df, ["Case", "Activity", "Timestamp", "Resource"])

    # Initialize required output columns
    df["error_detected"] = False  # bool
    df["detected_error_types"] = ""  # str

    # --- Pollution: compute base labels for all rows (used by distorted + synonymous)
    base_labels = []
    polluted_flags = []
    for a in df["Activity"].tolist():
        base, is_polluted = extract_base_label_and_polluted(a)
        base_labels.append(base)
        polluted_flags.append(is_polluted)

    df["_base_activity"] = pd.Series(base_labels, index=df.index)
    df["_is_polluted"] = pd.Series(polluted_flags, index=df.index)

    for idx, is_pol in df["_is_polluted"].items():
        if bool(is_pol):
            _add_error(df, idx, "polluted")

    # --- Form-based (timestamp collisions within case with different activities)
    form_idxs = detect_form_based(df, case_col="Case", act_col="Activity", ts_col="Timestamp")
    for idx in form_idxs:
        _add_error(df, idx, "form-based")

    # --- Collateral (bursts within case, same resource, <=5s)
    collateral_idxs = detect_collateral(df, case_col="Case", ts_col="Timestamp", res_col="Resource", window_seconds=5)
    for idx in collateral_idxs:
        _add_error(df, idx, "collateral")

    # --- Distorted (character-level) on base labels only (after pollution removal)
    distorted_map = detect_distortions(df["_base_activity"], min_sim=0.86, max_len_diff_ratio=0.5)
    for idx, base in df["_base_activity"].items():
        base = _normalize_spaces(str(base))
        if base in distorted_map:
            _add_error(df, idx, "distorted")

    # --- Synonymous (semantic) on base labels only (after pollution removal)
    synonym_map, _clusters = detect_synonyms(df["_base_activity"], sim_threshold=0.80)
    for idx, base in df["_base_activity"].items():
        base = _normalize_spaces(str(base))
        if base in synonym_map:
            _add_error(df, idx, "synonymous")

    # --- Homonymous (contextual divergence within case)
    homonymous_idxs = detect_homonymous(df, case_col="Case", act_col="Activity", ts_col="Timestamp")
    for idx in homonymous_idxs:
        _add_error(df, idx, "homonymous")

    # Ensure detected_error_types is pipe-separated in a stable order (optional but reproducible)
    order = ["form-based", "polluted", "distorted", "synonymous", "collateral", "homonymous"]
    order_index = {t: i for i, t in enumerate(order)}

    def _sort_types(s: str) -> str:
        if not s:
            return ""
        parts = [p for p in s.split("|") if p]
        parts = sorted(set(parts), key=lambda x: order_index.get(x, 999))
        return "|".join(parts)

    df["detected_error_types"] = df["detected_error_types"].astype(str).map(_sort_types)
    df["error_detected"] = df["error_detected"].astype(bool)

    # Drop internal helper columns
    df = df.drop(columns=["_base_activity", "_is_polluted"], errors="ignore")

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()