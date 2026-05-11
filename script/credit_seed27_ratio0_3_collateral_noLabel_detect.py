import sys
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Utilities
# -----------------------------
def _norm_space(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _safe_lower(s: str) -> str:
    return _norm_space(s).lower()


def _parse_timestamp_series(ts: pd.Series) -> pd.Series:
    # Robust parsing; keep NaT for unparseable
    return pd.to_datetime(ts, errors="coerce")


def _add_error(df: pd.DataFrame, idx, err_type: str):
    # Accumulate independently; keep deterministic ordering later
    df.at[idx, "_err_set"].add(err_type)


def _finalize_errors(df: pd.DataFrame):
    order = ["form-based", "polluted", "distorted", "synonymous", "collateral", "homonymous"]
    detected = []
    for s in df["_err_set"].tolist():
        if not s:
            detected.append("")
        else:
            detected.append("|".join([e for e in order if e in s]))
    df["detected_error_types"] = detected
    df["error_detected"] = df["detected_error_types"].astype(str).str.len().gt(0)
    df.drop(columns=["_err_set"], inplace=True)


# -----------------------------
# Polluted detection + base extraction
# -----------------------------
_SEP_CHARS = r"[_\-\.\|:/#]"
_BRACKETED_TOKEN = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")
_LONG_DIGITS = re.compile(r"\d{6,}")
_TS_FRAGMENT = re.compile(r"(19|20)\d{2}[01]\d[0-3]\d([ T]?\d{2}\d{2}\d{2}(\d{3,6})?)?")
_ALNUM_ID = re.compile(r"(?i)^[a-z0-9]{6,}$")
_RESOURCE_LIKE = re.compile(r"(?i)^(system|user|clerk|manager|nurse|doctor|worker|machine|staff|agent)[-_]?\d{3,}$")
_MIXED_ID = re.compile(r"(?i)^(?=.*[a-z])(?=.*\d)[a-z0-9]{5,}$")


def _token_is_machiney(tok: str) -> bool:
    t = _norm_space(tok)
    if not t:
        return False
    # Strip surrounding brackets for evaluation
    t_stripped = re.sub(r"^[\[\(\{]\s*|\s*[\]\)\}]$", "", t).strip()
    if not t_stripped:
        return False

    if _BRACKETED_TOKEN.match(t):
        # bracketed tokens are often IDs; check inside
        t_eval = t_stripped
    else:
        t_eval = t_stripped

    if _RESOURCE_LIKE.match(t_eval):
        return True
    if _TS_FRAGMENT.search(t_eval):
        return True
    if _LONG_DIGITS.search(t_eval):
        return True
    if _ALNUM_ID.match(t_eval):
        return True
    if _MIXED_ID.match(t_eval):
        return True
    # Very long token with few spaces is suspicious
    if len(t_eval) >= 12 and " " not in t_eval:
        # contains digits or mixed case
        if any(ch.isdigit() for ch in t_eval):
            return True
    return False


def extract_base_label(activity: str):
    """
    Returns (base_label, is_polluted).
    Heuristic: remove machine-generated prefix/suffix tokens separated by common separators or bracket groups.
    """
    raw = _norm_space(activity)
    if not raw:
        return "", False

    # If label is like "[Cook Dinner][BxA81Qe][214503882]" -> keep first bracket group as base
    bracket_groups = re.findall(r"\[[^\]]+\]", raw)
    if len(bracket_groups) >= 2:
        base = re.sub(r"^\[|\]$", "", bracket_groups[0]).strip()
        base = _norm_space(base)
        if base and base != raw:
            return base, True

    # Split by separators while keeping words intact
    parts = re.split(_SEP_CHARS, raw)
    parts = [p.strip() for p in parts if p is not None and p.strip() != ""]

    # Also consider colon/pipe separated segments that may not be captured if no separators
    # (already covered by _SEP_CHARS)

    # If no separators, try to detect trailing machiney chunk appended without separator:
    # e.g., "CookDinner74lwMHm20230929212246639"
    if len(parts) == 1:
        s = parts[0]
        # Find a suffix that looks like machiney: long digits or timestamp fragment or mixed id
        m = re.search(r"(?i)(.*?)(\d{6,}|(19|20)\d{2}[01]\d[0-3]\d.*)$", s)
        if m:
            base_candidate = _norm_space(m.group(1))
            suffix = m.group(2)
            if base_candidate and _token_is_machiney(suffix):
                # Insert spaces between camelcase as a mild normalization for base
                base_candidate = re.sub(r"([a-z])([A-Z])", r"\1 \2", base_candidate).strip()
                return base_candidate, True

        # Another pattern: base + mixed alnum id
        m2 = re.search(r"(?i)(.*?)([a-z0-9]{6,})$", s)
        if m2:
            base_candidate = _norm_space(m2.group(1))
            suffix = m2.group(2)
            if base_candidate and _token_is_machiney(suffix):
                base_candidate = re.sub(r"([a-z])([A-Z])", r"\1 \2", base_candidate).strip()
                return base_candidate, True

        return raw, False

    # If multiple parts, remove machiney tokens from start/end only (preserve meaningful middle)
    left = 0
    right = len(parts) - 1

    while left <= right and _token_is_machiney(parts[left]):
        left += 1
    while right >= left and _token_is_machiney(parts[right]):
        right -= 1

    base_parts = parts[left : right + 1]
    base = _norm_space(" ".join(base_parts))

    # If we removed something and base is non-empty and differs materially -> polluted
    removed = (left > 0) or (right < len(parts) - 1)
    if removed and base:
        return base, True

    # Special case: "Manager000001_Approve request" -> first token resource-like without separator split?
    # It will split into ["Manager000001", "Approve request"] and mark as polluted.
    return raw, False


# -----------------------------
# Distorted detection (edit similarity on cleaned labels)
# -----------------------------
def _levenshtein(a: str, b: str) -> int:
    # Iterative DP, O(min(len(a),len(b))) memory
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
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
    a = _norm_space(a)
    b = _norm_space(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _levenshtein(a.lower(), b.lower())
    denom = max(len(a), len(b))
    return 1.0 - (dist / denom)


def detect_distorted(df: pd.DataFrame, cleaned_col: str):
    # Build frequency on cleaned labels (after pollution removal)
    cleaned = df[cleaned_col].fillna("").map(_norm_space)
    freq = Counter(cleaned.tolist())

    labels = [l for l in freq.keys() if l]
    if len(labels) < 2:
        return {}

    # Candidate canonical labels: prefer frequent and "stable" (no weird spacing)
    # We do not assume frequent == clean; we only use frequency to break ties among very close strings.
    # We'll map each label to a more canonical close neighbor if similarity is high.
    mapping = {}  # label -> canonical_label

    # Precompute pairwise close neighbors using length-bounded comparisons to reduce cost
    labels_sorted = sorted(labels, key=lambda x: (len(x), -freq[x], x.lower()))
    for i, a in enumerate(labels_sorted):
        best = None
        best_sim = 0.0
        for j in range(len(labels_sorted)):
            if i == j:
                continue
            b = labels_sorted[j]
            # quick length filter
            la, lb = len(a), len(b)
            if abs(la - lb) > max(2, int(0.25 * max(la, lb))):
                continue
            sim = _edit_similarity(a, b)
            if sim >= 0.90:
                # choose canonical: higher freq; if tie, fewer "oddities"
                if best is None or sim > best_sim:
                    best = b
                    best_sim = sim
                elif abs(sim - best_sim) < 1e-9:
                    # tie-breaker by frequency
                    if freq[b] > freq[best]:
                        best = b
                        best_sim = sim

        if best is not None:
            # Decide direction: map less frequent to more frequent if very close
            if freq[a] < freq[best]:
                mapping[a] = best
            elif freq[a] == freq[best]:
                # If equal frequency, prefer the one with fewer internal spacing anomalies
                def oddity_score(s):
                    # penalize multiple spaces, split words like "Appr ove"
                    return int(bool(re.search(r"\b\w\s+\w\b", s))) + int("  " in s)

                if oddity_score(a) > oddity_score(best):
                    mapping[a] = best

    return mapping


# -----------------------------
# Synonymous detection (sentence-transformers on cleaned labels)
# -----------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

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

    def groups(self):
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return list(g.values())


def detect_synonyms(df: pd.DataFrame, cleaned_col: str, threshold: float = 0.80):
    cleaned = df[cleaned_col].fillna("").map(_norm_space)
    freq = Counter(cleaned.tolist())
    labels = [l for l in freq.keys() if l]
    if len(labels) < 2:
        return {}, {}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(labels, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    sim = util.cos_sim(embeddings, embeddings).cpu().numpy()

    uf = UnionFind(labels)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(labels[i], labels[j])

    clusters = [c for c in uf.groups() if len(c) >= 2]
    if not clusters:
        return {}, {}

    # canonical per cluster: most frequent; tie-breaker: shortest then lexicographic
    canonical = {}
    cluster_of = {}
    for c in clusters:
        c_sorted = sorted(c, key=lambda x: (-freq[x], len(x), x.lower()))
        canon = c_sorted[0]
        for member in c:
            canonical[member] = canon
            cluster_of[member] = tuple(sorted(c, key=lambda x: x.lower()))
    return canonical, cluster_of


# -----------------------------
# Form-based detection (timestamp collisions within case with different activities)
# -----------------------------
def detect_formbased(df: pd.DataFrame, case_col: str, act_col: str, ts_col: str):
    # Mark all events in a (case, timestamp) group if size>=2 and activities differ
    for (case, ts), g in df.groupby([case_col, ts_col], dropna=False, sort=False):
        if pd.isna(ts):
            continue
        if len(g) < 2:
            continue
        acts = g[act_col].fillna("").map(_norm_space)
        if acts.nunique(dropna=False) >= 2:
            for idx in g.index:
                _add_error(df, idx, "form-based")


# -----------------------------
# Collateral detection (bursts within case, consecutive, <=5s, same resource)
# -----------------------------
def detect_collateral(df: pd.DataFrame, case_col: str, ts_parsed_col: str, res_col: str, window_seconds: float = 5.0):
    # Work per case in time order; consecutive bursts with same resource and short gaps
    for case, g in df.groupby(case_col, sort=False):
        gg = g.copy()
        gg = gg.sort_values([ts_parsed_col, "_row_id"], kind="mergesort")
        idxs = gg.index.to_list()
        tss = gg[ts_parsed_col].to_list()
        ress = gg[res_col].fillna("").map(_norm_space).to_list()

        # Identify burst segments
        start = 0
        while start < len(idxs):
            end = start
            # Expand while consecutive within window and same resource (non-empty)
            while end + 1 < len(idxs):
                t1, t2 = tss[end], tss[end + 1]
                if pd.isna(t1) or pd.isna(t2):
                    break
                dt = (t2 - t1).total_seconds()
                same_res = (ress[end] != "" and ress[end] == ress[end + 1])
                if (dt <= window_seconds) and same_res:
                    end += 1
                else:
                    break

            seg_len = end - start + 1
            if seg_len >= 2:
                for k in range(start, end + 1):
                    _add_error(df, idxs[k], "collateral")
            start = end + 1


# -----------------------------
# Homonymous detection (exactly 2 occurrences in a case with different immediate predecessor)
# Prefer collateral when occurrences >=3 (do not flag homonymous then)
# -----------------------------
def detect_homonymous(df: pd.DataFrame, case_col: str, act_col: str, ts_parsed_col: str):
    for case, g in df.groupby(case_col, sort=False):
        gg = g.copy()
        gg = gg.sort_values([ts_parsed_col, "_row_id"], kind="mergesort")
        acts = gg[act_col].fillna("").map(_norm_space).to_list()
        idxs = gg.index.to_list()

        positions = defaultdict(list)
        for pos, a in enumerate(acts):
            if a:
                positions[a].append(pos)

        for a, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            if len(pos_list) >= 3:
                # Prefer collateral over homonymous in ambiguous repeated cases
                continue
            # Exactly 2 occurrences
            p1, p2 = pos_list
            prev1 = acts[p1 - 1] if p1 - 1 >= 0 else ""
            prev2 = acts[p2 - 1] if p2 - 1 >= 0 else ""
            if prev1 and prev2 and prev1 != prev2:
                _add_error(df, idxs[p1], "homonymous")
                _add_error(df, idxs[p2], "homonymous")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python script.py input.csv output.csv")

    inp = sys.argv[1]
    out = sys.argv[2]

    df = pd.read_csv(inp)

    required = ["Case", "Activity", "Timestamp", "Resource"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Initialize required output columns (exact names)
    df["error_detected"] = False
    df["detected_error_types"] = ""
    df["_err_set"] = [set() for _ in range(len(df))]

    # Stable row id for deterministic ordering within same timestamp
    df["_row_id"] = np.arange(len(df), dtype=int)

    # Parse timestamps
    df["_ts_parsed"] = _parse_timestamp_series(df["Timestamp"])

    # Cleaned activity (pollution removed)
    base_labels = []
    polluted_flags = []
    for a in df["Activity"].tolist():
        base, is_pol = extract_base_label(a)
        base_labels.append(base)
        polluted_flags.append(is_pol)
    df["_activity_base"] = base_labels
    df["_is_polluted"] = polluted_flags

    # 1) Form-based (independent)
    detect_formbased(df, "Case", "Activity", "Timestamp")

    # 2) Polluted (independent)
    for idx, is_pol in zip(df.index, df["_is_polluted"].tolist()):
        if is_pol:
            _add_error(df, idx, "polluted")

    # 3) Distorted (on base labels only; independent)
    distortion_map = detect_distorted(df, "_activity_base")
    if distortion_map:
        for idx, base in zip(df.index, df["_activity_base"].tolist()):
            base = _norm_space(base)
            if base in distortion_map:
                _add_error(df, idx, "distorted")

    # 4) Synonymous (on base labels only; independent)
    synonym_canon, _ = detect_synonyms(df, "_activity_base", threshold=0.80)
    if synonym_canon:
        for idx, base in zip(df.index, df["_activity_base"].tolist()):
            base = _norm_space(base)
            if base and base in synonym_canon and synonym_canon[base] != base:
                _add_error(df, idx, "synonymous")

    # 5) Collateral (independent)
    detect_collateral(df, "Case", "_ts_parsed", "Resource", window_seconds=5.0)

    # 6) Homonymous (independent; with preference rule)
    detect_homonymous(df, "Case", "Activity", "_ts_parsed")

    # Finalize required columns
    _finalize_errors(df)

    # Drop internal helper columns
    df.drop(columns=["_row_id", "_ts_parsed", "_activity_base", "_is_polluted"], inplace=True, errors="ignore")

    # Save
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()