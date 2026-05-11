import sys
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Utilities
# -----------------------------
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _add_error(errors_by_idx: dict, idx: int, err_type: str):
    errors_by_idx[idx].add(err_type)


def _finalize_errors(df: pd.DataFrame, errors_by_idx: dict):
    df["detected_error_types"] = ""
    df["error_detected"] = False

    # stable ordering
    order = ["form-based", "polluted", "distorted", "synonymous", "collateral", "homonymous"]

    for idx, errs in errors_by_idx.items():
        if not errs:
            continue
        df.at[idx, "error_detected"] = True
        df.at[idx, "detected_error_types"] = "|".join([e for e in order if e in errs])


# -----------------------------
# Polluted detection + base label extraction
# -----------------------------
_SEP_CHARS = r"[_\-\.\|:/#\\]"
_BRACKETED_TOKEN = re.compile(r"(\[[^\]]{2,}\]|\([^\)]{2,}\))")
_LONG_DIGITS = re.compile(r"\d{6,}")
_DATEISH = re.compile(r"(19|20)\d{2}[01]\d[0-3]\d")  # yyyymmdd
_TIMEISH = re.compile(r"\b[0-2]\d[0-5]\d[0-5]\d\b")  # hhmmss
_ISO_T = re.compile(r"(19|20)\d{2}[01]\d[0-3]\dT[0-2]\d[0-5]\d[0-5]\d")
_MIXED_ALNUM = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{6,}")
_RESOURCEISH = re.compile(r"^[A-Za-z]+-?\d{3,}$")  # e.g., Manager-000003, Manager000003
_HEXISH = re.compile(r"\b[0-9a-fA-F]{8,}\b")


def _token_looks_machine_generated(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False

    # remove surrounding brackets/parentheses for evaluation
    t2 = re.sub(r"^[\[\(]\s*|\s*[\]\)]$", "", t).strip()

    if _RESOURCEISH.match(t2):
        return True
    if _ISO_T.search(t2):
        return True
    if _DATEISH.search(t2) and (_TIMEISH.search(t2) or _LONG_DIGITS.search(t2)):
        return True
    if _LONG_DIGITS.search(t2):
        return True
    if _MIXED_ALNUM.search(t2):
        return True
    if _HEXISH.search(t2):
        return True

    # timestamp fragments like 20230929 130852312000 or 133002114000
    if re.search(r"\b\d{8}\b", t2) and re.search(r"\b\d{9,}\b", t2):
        return True

    # very long token with many digits
    if len(t2) >= 12 and sum(ch.isdigit() for ch in t2) >= 6:
        return True

    return False


def extract_base_label_and_polluted(activity: str):
    """
    Returns: (base_label, is_polluted)
    base_label is a cleaned label with likely machine-generated affixes removed.
    """
    raw = _norm_ws(_safe_str(activity))
    if not raw:
        return raw, False

    # Strategy:
    # 1) If bracketed segments exist and at least one looks machine-generated, drop those segments.
    # 2) Split by separators and whitespace; remove leading/trailing machine tokens.
    # 3) Also handle concatenations like CookDinner74lwMHm20230929212246639 by splitting trailing machine tail.
    s = raw

    # 1) Remove bracketed machine tokens
    bracketed = _BRACKETED_TOKEN.findall(s)
    removed_any = False
    if bracketed:
        for b in bracketed:
            if _token_looks_machine_generated(b):
                s = s.replace(b, " ")
                removed_any = True
        s = _norm_ws(s)

    # 2) Split by separators into parts, but keep meaningful middle phrase.
    # We'll remove machine-looking tokens from start/end only (conservative).
    parts = re.split(_SEP_CHARS, s)
    parts = [_norm_ws(p) for p in parts if _norm_ws(p)]
    if len(parts) >= 2:
        # remove machine tokens at ends
        i, j = 0, len(parts) - 1
        while i <= j and _token_looks_machine_generated(parts[i]):
            i += 1
            removed_any = True
        while j >= i and _token_looks_machine_generated(parts[j]):
            j -= 1
            removed_any = True
        core_parts = parts[i : j + 1]
        if core_parts:
            s2 = _norm_ws(" ".join(core_parts))
        else:
            s2 = _norm_ws(s)
    else:
        s2 = _norm_ws(s)

    # 3) Handle prefix like Manager000001_Approve request
    # If first token is resource-ish and rest looks like words, drop it.
    toks = s2.split()
    if len(toks) >= 2 and _RESOURCEISH.match(toks[0]):
        s2 = _norm_ws(" ".join(toks[1:]))
        removed_any = True

    # 4) Handle concatenated tail: letters+digits+mixed tail appended to last word
    # Example: CookDinner74lwMHm20230929212246639 -> Cook Dinner
    # We'll try to split camelcase first, then remove trailing machine tail.
    def split_camel(text: str) -> str:
        # Insert space between lower->upper and letter->digit boundaries
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
        return _norm_ws(text)

    s3 = split_camel(s2)

    # If last token looks machine-generated and preceding tokens exist, drop it
    toks3 = s3.split()
    if len(toks3) >= 2 and _token_looks_machine_generated(toks3[-1]):
        s3 = _norm_ws(" ".join(toks3[:-1]))
        removed_any = True

    # If still contains a token with long mixed tail, trim within token
    # e.g., "CookDinner74lwMHm20230929212246639" as single token
    if len(toks3) == 1:
        t = toks3[0]
        # split at first occurrence of a long machine-ish tail
        m = re.search(r"([A-Za-z][A-Za-z ]{2,}?)([0-9A-Za-z]{6,}\d{6,}.*)$", t)
        if m:
            head = m.group(1)
            head = split_camel(head)
            if head and head != t:
                s3 = _norm_ws(head)
                removed_any = True

    base = _norm_ws(s3)
    is_polluted = removed_any and base and base.lower() != raw.lower()

    return base if base else raw, bool(is_polluted)


# -----------------------------
# Distorted detection (character-level)
# -----------------------------
def _levenshtein_distance(a: str, b: str) -> int:
    # iterative DP, O(len(a)*len(b))
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
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
    a = a.lower()
    b = b.lower()
    if a == b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    d = _levenshtein_distance(a, b)
    return 1.0 - (d / m)


def detect_distortions(clean_labels: pd.Series, label_counts: Counter):
    """
    Returns mapping: label -> canonical_label (for distorted labels only)
    Uses only cleaned labels (pollution removed).
    """
    unique = list(label_counts.keys())
    if len(unique) <= 1:
        return {}

    # Build candidate pairs by length proximity to reduce comparisons
    by_len = defaultdict(list)
    for lab in unique:
        by_len[len(lab)].append(lab)

    distorted_to_canon = {}

    # thresholds (conservative)
    # - require high similarity
    # - require small absolute edit distance
    for lab in unique:
        best = None
        best_sim = -1.0
        best_dist = None

        L = len(lab)
        candidates = []
        for l2 in range(max(1, L - 3), L + 4):
            candidates.extend(by_len.get(l2, []))

        for cand in candidates:
            if cand == lab:
                continue
            # quick reject: if first letters very different and short, skip
            if L <= 6 and cand and lab and cand[0].lower() != lab[0].lower():
                continue

            sim = _edit_similarity(lab, cand)
            if sim < 0.90:
                continue
            dist = _levenshtein_distance(lab.lower(), cand.lower())

            # require small edit distance relative to length
            if dist > 3 and dist > int(0.12 * max(len(lab), len(cand))):
                continue

            # prefer more frequent and slightly longer/more "stable" (fewer non-letters)
            if sim > best_sim:
                best = cand
                best_sim = sim
                best_dist = dist
            elif sim == best_sim and best is not None:
                # tie-breaker: higher frequency
                if label_counts[cand] > label_counts[best]:
                    best = cand
                    best_dist = dist

        if best is None:
            continue

        # Decide canonical: prefer the more frequent label as canonical
        if label_counts[best] > label_counts[lab]:
            canon = best
            var = lab
        elif label_counts[best] < label_counts[lab]:
            # if lab is more frequent, treat best as distorted instead (skip mapping for lab)
            continue
        else:
            # equal frequency: prefer label with fewer non-alphabetic chars and fewer double spaces
            def stability_score(x: str) -> tuple:
                non_alpha = sum(0 if ch.isalpha() or ch.isspace() else 1 for ch in x)
                multi_space = 1 if "  " in x else 0
                return (non_alpha, multi_space, len(x))

            if stability_score(best) < stability_score(lab):
                canon = best
                var = lab
            else:
                continue

        # Avoid mapping if they are identical ignoring spaces/case (that's not distortion)
        if re.sub(r"\s+", "", var).lower() == re.sub(r"\s+", "", canon).lower():
            # still could be internal word split/extra spaces -> distortion, allow it
            pass

        distorted_to_canon[var] = canon

    return distorted_to_canon


# -----------------------------
# Synonymous detection (semantic)
# -----------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

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

    def components(self):
        comps = defaultdict(list)
        for x in self.parent:
            comps[self.find(x)].append(x)
        return list(comps.values())


def detect_synonyms(clean_labels: pd.Series, label_counts: Counter, threshold: float = 0.80):
    """
    Returns mapping: label -> canonical_label (for non-canonical synonyms only)
    Uses sentence-transformers on unique cleaned labels.
    """
    unique = list(label_counts.keys())
    if len(unique) <= 1:
        return {}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique, convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(embeddings, embeddings).cpu().numpy()

    uf = UnionFind(unique)
    n = len(unique)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(unique[i], unique[j])

    clusters = [c for c in uf.components() if len(c) >= 2]
    if not clusters:
        return {}

    synonym_to_canon = {}
    for cluster in clusters:
        # canonical = most frequent label in dataset; tie-breaker: shortest then lexicographic
        cluster_sorted = sorted(
            cluster,
            key=lambda x: (-label_counts[x], len(x), x.lower()),
        )
        canon = cluster_sorted[0]
        for lab in cluster:
            if lab != canon:
                synonym_to_canon[lab] = canon

    return synonym_to_canon


# -----------------------------
# Form-based detection (timestamp collisions within case)
# -----------------------------
def detect_form_based(df: pd.DataFrame, errors_by_idx: dict):
    # within each case, same timestamp, >=2 events, different activity labels
    for case_id, g in df.groupby("Case", sort=False):
        # group by exact timestamp string (after parsing, use normalized)
        for ts, gg in g.groupby("_ts_norm", sort=False):
            if len(gg) < 2:
                continue
            acts = gg["Activity"].astype(str).tolist()
            if len(set(acts)) >= 2:
                for idx in gg.index:
                    _add_error(errors_by_idx, idx, "form-based")


# -----------------------------
# Collateral detection (bursts within case)
# -----------------------------
def detect_collateral(df: pd.DataFrame, errors_by_idx: dict, window_seconds: float = 5.0):
    # For each case, sort by timestamp; find consecutive runs where dt<=5s or identical timestamps
    for case_id, g in df.groupby("Case", sort=False):
        gg = g.sort_values(["_ts", "_row_order"], kind="mergesort")
        idxs = gg.index.to_list()
        ts = gg["_ts"].to_numpy()
        res = gg["Resource"].astype(str).to_list()

        # Build bursts: consecutive events with dt<=window and same resource (strong contextual similarity)
        start = 0
        while start < len(idxs):
            end = start
            while end + 1 < len(idxs):
                dt = (ts[end + 1] - ts[end]) / np.timedelta64(1, "s")
                if np.isnan(dt):
                    break
                if dt <= window_seconds and res[end + 1] == res[end] and _safe_str(res[end]).strip() != "":
                    end += 1
                else:
                    break

            if end - start + 1 >= 2:
                for k in range(start, end + 1):
                    _add_error(errors_by_idx, idxs[k], "collateral")

            start = end + 1


# -----------------------------
# Homonymous detection (same label, different preceding context within case)
# -----------------------------
def detect_homonymous(df: pd.DataFrame, errors_by_idx: dict):
    # Prefer collateral if occurrences are 3+; only consider exactly 2 occurrences with different preceding activity.
    for case_id, g in df.groupby("Case", sort=False):
        gg = g.sort_values(["_ts", "_row_order"], kind="mergesort")
        acts = gg["_clean_activity"].tolist()
        idxs = gg.index.tolist()

        positions_by_label = defaultdict(list)
        for pos, lab in enumerate(acts):
            positions_by_label[lab].append(pos)

        for lab, poss in positions_by_label.items():
            if len(poss) != 2:
                continue  # per instruction: prefer homonymous only if exactly 2 occurrences
            p1, p2 = poss
            if p1 == 0 or p2 == 0:
                continue  # no preceding context for first event
            prev1 = acts[p1 - 1]
            prev2 = acts[p2 - 1]
            if prev1 != prev2:
                # If both occurrences are already collateral, do not add homonymous (collateral preferred)
                idx1 = idxs[p1]
                idx2 = idxs[p2]
                if ("collateral" in errors_by_idx[idx1]) and ("collateral" in errors_by_idx[idx2]):
                    continue
                _add_error(errors_by_idx, idx1, "homonymous")
                _add_error(errors_by_idx, idx2, "homonymous")


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

    # Preserve original row order for stable sorting ties
    df["_row_order"] = np.arange(len(df), dtype=int)

    # Parse timestamps robustly
    df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    # normalized string for exact collision grouping (keep original precision if possible)
    # Use original Timestamp string trimmed; if NaT, keep empty
    df["_ts_norm"] = df["Timestamp"].apply(lambda x: _norm_ws(_safe_str(x)))

    # Initialize error accumulator
    errors_by_idx = defaultdict(set)

    # --- Polluted + cleaned activity
    base_labels = []
    polluted_flags = []
    for a in df["Activity"].tolist():
        base, is_pol = extract_base_label_and_polluted(a)
        base_labels.append(base)
        polluted_flags.append(is_pol)

    df["_clean_activity"] = base_labels

    for idx, is_pol in zip(df.index, polluted_flags):
        if is_pol:
            _add_error(errors_by_idx, idx, "polluted")

    # --- Form-based
    detect_form_based(df, errors_by_idx)

    # --- Collateral
    detect_collateral(df, errors_by_idx, window_seconds=5.0)

    # --- Distorted (on cleaned labels only)
    clean_counts = Counter(df["_clean_activity"].astype(str).apply(_norm_ws).tolist())
    # normalize cleaned labels in df to match Counter keys
    df["_clean_activity"] = df["_clean_activity"].astype(str).apply(_norm_ws)

    distorted_map = detect_distortions(df["_clean_activity"], clean_counts)
    if distorted_map:
        for idx, lab in zip(df.index, df["_clean_activity"].tolist()):
            if lab in distorted_map:
                _add_error(errors_by_idx, idx, "distorted")

    # --- Synonymous (on cleaned labels only)
    synonym_map = detect_synonyms(df["_clean_activity"], clean_counts, threshold=0.80)
    if synonym_map:
        for idx, lab in zip(df.index, df["_clean_activity"].tolist()):
            if lab in synonym_map:
                _add_error(errors_by_idx, idx, "synonymous")

    # --- Homonymous (on cleaned labels only)
    detect_homonymous(df, errors_by_idx)

    # Finalize required output columns
    _finalize_errors(df, errors_by_idx)

    # Drop internal columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # Ensure exact column names exist and types are correct
    if "error_detected" not in df.columns or "detected_error_types" not in df.columns:
        raise RuntimeError("Failed to create required output columns.")

    # Enforce boolean dtype for error_detected
    df["error_detected"] = df["error_detected"].astype(bool)
    df["detected_error_types"] = df["detected_error_types"].astype(str)

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()