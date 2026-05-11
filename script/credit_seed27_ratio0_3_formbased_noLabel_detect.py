import sys
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Utilities
# -----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def parse_timestamp_series(ts: pd.Series) -> pd.Series:
    # Robust parsing; keep NaT for unparseable
    return pd.to_datetime(ts, errors="coerce")


def add_error(errors_by_idx: Dict[int, Set[str]], idx: int, err_type: str):
    errors_by_idx[idx].add(err_type)


def finalize_errors(df: pd.DataFrame, errors_by_idx: Dict[int, Set[str]]) -> pd.DataFrame:
    detected = []
    types = []
    for i in range(len(df)):
        errs = sorted(errors_by_idx.get(i, set()))
        if errs:
            detected.append(True)
            types.append("|".join(errs))
        else:
            detected.append(False)
            types.append("")
    df["error_detected"] = detected
    df["detected_error_types"] = types
    return df


# -----------------------------
# Pollution detection / cleaning
# -----------------------------
POLLUTION_SEP_CHARS = r"[_\-\.\|:/#]"
BRACKET_CHARS = r"[\[\]\(\)\{\}]"

# Heuristics for "machine-generated" tokens
RE_LONG_DIGITS = re.compile(r"\d{6,}")
RE_MIXED_ALNUM = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{6,}")
RE_TIMESTAMPISH = re.compile(
    r"(?i)\b(?:19|20)\d{2}[01]\d[0-3]\d(?:[ T]?[0-2]\d[0-5]\d[0-5]\d(?:\d{3,6})?)?\b"
)
RE_RESOURCEISH = re.compile(r"(?i)\b(?:system|clerk|manager|worker|nurse|doctor|staff|machine)[-_]?\d{3,}\b")
RE_HEXISH = re.compile(r"(?i)\b[0-9a-f]{8,}\b")


def token_is_machiney(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    # Remove surrounding brackets
    t2 = re.sub(rf"^{BRACKET_CHARS}+|{BRACKET_CHARS}+$", "", t).strip()
    if not t2:
        return False

    # Strong signals
    if RE_LONG_DIGITS.search(t2):
        return True
    if RE_TIMESTAMPISH.search(t2):
        return True
    if RE_MIXED_ALNUM.fullmatch(t2):
        return True
    if RE_RESOURCEISH.search(t2):
        return True
    if RE_HEXISH.fullmatch(t2):
        return True

    # Very long "id-like" token
    if len(t2) >= 12 and re.fullmatch(r"[A-Za-z0-9]+", t2):
        # if it has low vowel ratio, often random
        vowels = sum(ch.lower() in "aeiou" for ch in t2)
        if vowels / max(1, len(t2)) < 0.25:
            return True

    return False


def split_activity_tokens(activity: str) -> List[str]:
    s = normalize_ws(activity)
    # Split on separators and whitespace, but keep meaningful words
    parts = re.split(rf"(?:\s+|{POLLUTION_SEP_CHARS}+)", s)
    parts = [p for p in (p.strip() for p in parts) if p]
    return parts


def remove_bracketed_machine_tokens(activity: str) -> Tuple[str, bool]:
    """
    Remove bracketed segments that look machine-generated, e.g. [BxA81Qe], (80nSeSg), [214503882]
    Keep bracketed segments that look human (rare in these logs, but be conservative).
    """
    s = activity
    polluted = False

    # Replace bracketed groups iteratively
    pattern = re.compile(r"(\[[^\]]+\]|\([^)]+\)|\{[^}]+\})")
    out = []
    last = 0
    for m in pattern.finditer(s):
        out.append(s[last:m.start()])
        content = m.group(0)[1:-1].strip()
        if token_is_machiney(content):
            polluted = True
            # drop it
        else:
            out.append(m.group(0))
        last = m.end()
    out.append(s[last:])
    cleaned = normalize_ws("".join(out))
    return cleaned, polluted


def detect_and_clean_pollution(activity: str) -> Tuple[str, bool]:
    """
    Returns (cleaned_activity, is_polluted).
    Cleaning removes only machine-generated tokens/prefixes/suffixes when separable.
    """
    raw = normalize_ws(activity)
    if not raw:
        return raw, False

    # First remove bracketed machine tokens
    s, polluted_brackets = remove_bracketed_machine_tokens(raw)

    # Token-based removal for prefix/suffix machine tokens
    tokens = split_activity_tokens(s)
    if not tokens:
        return s, polluted_brackets

    # Identify machiney tokens
    machine_flags = [token_is_machiney(t) for t in tokens]

    # Remove leading machine tokens
    start = 0
    while start < len(tokens) and machine_flags[start]:
        start += 1

    # Remove trailing machine tokens
    end = len(tokens)
    while end > start and machine_flags[end - 1]:
        end -= 1

    # If we removed something, mark polluted
    polluted_tokens = (start > 0) or (end < len(tokens))

    # Reconstruct cleaned label from remaining tokens, but preserve original spacing as best-effort
    cleaned_tokens = tokens[start:end]
    cleaned = normalize_ws(" ".join(cleaned_tokens)) if cleaned_tokens else normalize_ws(s)

    # Additional heuristic: if raw contains separators and a machiney token anywhere, likely polluted
    any_machine = any(machine_flags)
    has_sep = bool(re.search(rf"{POLLUTION_SEP_CHARS}", raw)) or bool(re.search(rf"{BRACKET_CHARS}", raw))
    polluted_anywhere = has_sep and any_machine

    is_polluted = polluted_brackets or polluted_tokens or polluted_anywhere

    # Guardrail: don't "clean" into empty
    if not cleaned:
        cleaned = normalize_ws(s)

    return cleaned, is_polluted


# -----------------------------
# Distortion detection (character-level)
# -----------------------------
def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure a is shorter
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for i, chb in enumerate(b, start=1):
        cur = [i]
        for j, cha in enumerate(a, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if cha == chb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    a2 = a.lower()
    b2 = b.lower()
    if not a2 and not b2:
        return 1.0
    dist = levenshtein_distance(a2, b2)
    denom = max(len(a2), len(b2), 1)
    return 1.0 - (dist / denom)


def build_canonical_pool(clean_labels: pd.Series) -> Tuple[List[str], Dict[str, int]]:
    """
    Build canonical candidates from cleaned labels only.
    Prefer more frequent and more stable labels when labels are very close.
    """
    freq = Counter(clean_labels.tolist())
    labels = list(freq.keys())

    # Sort by frequency desc, then length desc (slightly favors fuller labels)
    labels_sorted = sorted(labels, key=lambda x: (freq[x], len(x)), reverse=True)

    canonicals: List[str] = []
    assigned_to: Dict[str, str] = {}

    # Threshold for "very close" at character level
    close_thr = 0.92

    for lab in labels_sorted:
        if lab in assigned_to:
            continue
        # Try to match to an existing canonical
        best_c = None
        best_sim = -1.0
        for c in canonicals:
            sim = edit_similarity(lab, c)
            if sim > best_sim:
                best_sim = sim
                best_c = c
        if best_c is not None and best_sim >= close_thr:
            assigned_to[lab] = best_c
        else:
            canonicals.append(lab)
            assigned_to[lab] = lab

    # Now, for each label, map to its canonical representative
    canonical_map = {}
    for lab in labels:
        # Find best canonical by similarity; if very close, map; else itself
        best_c = lab
        best_sim = 1.0
        for c in canonicals:
            sim = edit_similarity(lab, c)
            if sim > best_sim:
                best_sim = sim
                best_c = c
        if best_sim >= close_thr:
            canonical_map[lab] = best_c
        else:
            canonical_map[lab] = lab

    return canonicals, {k: freq[k] for k in freq}, canonical_map


def detect_distorted_rows(
    df: pd.DataFrame,
    cleaned_col: str,
    errors_by_idx: Dict[int, Set[str]],
):
    clean_labels = df[cleaned_col].fillna("").map(normalize_ws)
    _, freq, canonical_map = build_canonical_pool(clean_labels)

    # Distortion threshold: close but not identical
    sim_thr = 0.88

    for idx, lab in enumerate(clean_labels.tolist()):
        if not lab:
            continue
        canon = canonical_map.get(lab, lab)
        if canon == lab:
            continue

        sim = edit_similarity(lab, canon)
        # Only flag if sufficiently close (typo-like), and canonical is at least as frequent
        if sim >= sim_thr and freq.get(canon, 0) >= freq.get(lab, 0):
            add_error(errors_by_idx, idx, "distorted")


# -----------------------------
# Form-based detection (timestamp collisions within case)
# -----------------------------
def detect_form_based(
    df: pd.DataFrame,
    case_col: str,
    act_col: str,
    ts_parsed_col: str,
    errors_by_idx: Dict[int, Set[str]],
):
    # Group by case and exact timestamp; cluster size >=2 and different activities
    gcols = [case_col, ts_parsed_col]
    # Ignore NaT timestamps
    valid = df[ts_parsed_col].notna()
    sub = df.loc[valid, [case_col, act_col, ts_parsed_col]].copy()
    sub["_idx"] = df.index[valid].to_numpy()

    grouped = sub.groupby(gcols, dropna=True)
    for (_, _), grp in grouped:
        if len(grp) < 2:
            continue
        acts = grp[act_col].astype(str).tolist()
        if len(set(acts)) < 2:
            continue
        for ridx in grp["_idx"].tolist():
            add_error(errors_by_idx, int(ridx), "form-based")


# -----------------------------
# Collateral detection (bursts within case)
# -----------------------------
def detect_collateral(
    df: pd.DataFrame,
    case_col: str,
    ts_parsed_col: str,
    res_col: str,
    errors_by_idx: Dict[int, Set[str]],
    window_seconds: float = 5.0,
):
    # Sort within case by timestamp; consider consecutive events within <=5s and same resource (strong context)
    # If timestamp missing, skip those rows for collateral logic.
    df_work = df.copy()
    df_work["_idx"] = np.arange(len(df_work))
    df_work["_ts"] = df_work[ts_parsed_col]
    df_work["_res"] = df_work[res_col].map(safe_str)

    for case_id, grp in df_work.groupby(case_col, sort=False):
        g = grp.loc[grp["_ts"].notna()].sort_values("_ts", kind="mergesort")
        if len(g) < 2:
            continue

        idxs = g["_idx"].to_numpy()
        tss = g["_ts"].to_numpy()
        ress = g["_res"].to_numpy()

        # Identify burst segments
        start = 0
        for i in range(1, len(g)):
            dt = (pd.Timestamp(tss[i]) - pd.Timestamp(tss[i - 1])).total_seconds()
            same_ts = dt == 0
            close = dt <= window_seconds
            same_res = (ress[i] != "" and ress[i] == ress[i - 1])

            if (same_ts or close) and same_res:
                # continue burst
                continue
            else:
                # close segment [start, i-1]
                if i - start >= 2:
                    for j in range(start, i):
                        add_error(errors_by_idx, int(idxs[j]), "collateral")
                start = i

        # last segment
        if len(g) - start >= 2:
            for j in range(start, len(g)):
                add_error(errors_by_idx, int(idxs[j]), "collateral")


# -----------------------------
# Synonymous detection (semantic similarity on cleaned labels)
# -----------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
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


def detect_synonymous(
    df: pd.DataFrame,
    cleaned_col: str,
    errors_by_idx: Dict[int, Set[str]],
    sim_threshold: float = 0.80,
):
    cleaned = df[cleaned_col].fillna("").map(normalize_ws)
    # Unique non-empty labels
    uniq = sorted({x for x in cleaned.tolist() if x})
    if len(uniq) < 2:
        return

    # Frequency in dataset (by cleaned label)
    freq = Counter(cleaned.tolist())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(uniq, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    n = len(uniq)
    uf = UnionFind(n)

    # Union pairs above threshold (excluding i==j)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= sim_threshold:
                uf.union(i, j)

    # Build clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # Determine canonical per cluster (most frequent; tie-breaker: longer label)
    canonical_by_label: Dict[str, str] = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        member_labels = [uniq[i] for i in members]
        canonical = sorted(member_labels, key=lambda x: (freq.get(x, 0), len(x)), reverse=True)[0]
        for lab in member_labels:
            canonical_by_label[lab] = canonical

    # Flag rows whose cleaned label is in a synonym cluster but not canonical
    for idx, lab in enumerate(cleaned.tolist()):
        if not lab:
            continue
        canon = canonical_by_label.get(lab)
        if canon and canon != lab:
            add_error(errors_by_idx, idx, "synonymous")


# -----------------------------
# Homonymous detection (same label, different preceding activity within case)
# -----------------------------
def detect_homonymous(
    df: pd.DataFrame,
    case_col: str,
    cleaned_col: str,
    ts_parsed_col: str,
    errors_by_idx: Dict[int, Set[str]],
):
    df_work = df.copy()
    df_work["_idx"] = np.arange(len(df_work))
    df_work["_ts"] = df_work[ts_parsed_col]
    df_work["_act"] = df_work[cleaned_col].fillna("").map(normalize_ws)

    for case_id, grp in df_work.groupby(case_col, sort=False):
        # Order by timestamp when available; stable sort to preserve input order for ties/NaT
        g = grp.sort_values(["_ts"], kind="mergesort", na_position="last")
        acts = g["_act"].tolist()
        idxs = g["_idx"].tolist()

        # Map label -> list of (pos, preceding_label)
        occ: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for pos, lab in enumerate(acts):
            if not lab:
                continue
            prev = acts[pos - 1] if pos - 1 >= 0 else ""
            occ[lab].append((pos, prev))

        for lab, occurrences in occ.items():
            if len(occurrences) < 2:
                continue

            # Prefer collateral if 3+ occurrences (instruction)
            if len(occurrences) >= 3:
                continue

            # Exactly 2 occurrences: check preceding context differs and is non-empty
            (p1, prev1), (p2, prev2) = occurrences[0], occurrences[1]
            prev1 = normalize_ws(prev1)
            prev2 = normalize_ws(prev2)

            if prev1 and prev2 and prev1.lower() != prev2.lower():
                # Flag both occurrences as homonymous
                add_error(errors_by_idx, int(idxs[p1]), "homonymous")
                add_error(errors_by_idx, int(idxs[p2]), "homonymous")


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python script.py input.csv output.csv")

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv(in_path)

    required = ["Case", "Activity", "Timestamp", "Resource"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Initialize outputs
    df["error_detected"] = False
    df["detected_error_types"] = ""

    errors_by_idx: Dict[int, Set[str]] = defaultdict(set)

    # Parse timestamps
    df["_ts_parsed"] = parse_timestamp_series(df["Timestamp"])

    # Pollution detection + cleaned label
    cleaned_labels = []
    polluted_flags = []
    for a in df["Activity"].map(safe_str).tolist():
        cleaned, is_polluted = detect_and_clean_pollution(a)
        cleaned_labels.append(cleaned)
        polluted_flags.append(is_polluted)

    df["_activity_clean"] = pd.Series(cleaned_labels, index=df.index)
    df["_is_polluted"] = pd.Series(polluted_flags, index=df.index)

    for idx, is_pol in enumerate(polluted_flags):
        if is_pol:
            add_error(errors_by_idx, idx, "polluted")

    # Form-based (timestamp collisions within case with different activities)
    detect_form_based(df, "Case", "Activity", "_ts_parsed", errors_by_idx)

    # Collateral (bursts within case, same resource, <=5s)
    detect_collateral(df, "Case", "_ts_parsed", "Resource", errors_by_idx, window_seconds=5.0)

    # Distorted (character-level) on cleaned labels only
    detect_distorted_rows(df, "_activity_clean", errors_by_idx)

    # Synonymous (semantic) on cleaned labels only
    detect_synonymous(df, "_activity_clean", errors_by_idx, sim_threshold=0.80)

    # Homonymous (contextual path difference) on cleaned labels only
    detect_homonymous(df, "Case", "_activity_clean", "_ts_parsed", errors_by_idx)

    # Finalize required columns
    df = finalize_errors(df, errors_by_idx)

    # Drop internal columns
    df = df.drop(columns=[c for c in ["_ts_parsed", "_activity_clean", "_is_polluted"] if c in df.columns])

    # Save
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()