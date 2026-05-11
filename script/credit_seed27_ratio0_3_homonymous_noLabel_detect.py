import sys
import re
from collections import Counter, defaultdict

import numpy as np
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


def _norm_space(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _safe_lower(s: str) -> str:
    return _norm_space(s).lower()


def add_error(df: pd.DataFrame, idx, err_type: str):
    """Accumulate error types per row; keep deterministic ordering later."""
    if err_type not in VALID_ERROR_TYPES:
        return
    df.at[idx, "_err_set"].add(err_type)


def finalize_errors(df: pd.DataFrame):
    order = ["form-based", "polluted", "distorted", "synonymous", "collateral", "homonymous"]
    detected = []
    types = []
    for s in df["_err_set"].tolist():
        if not s:
            detected.append(False)
            types.append("")
        else:
            detected.append(True)
            types.append("|".join([t for t in order if t in s]))
    df["error_detected"] = detected
    df["detected_error_types"] = types
    df.drop(columns=["_err_set"], inplace=True)


# -----------------------------
# Polluted detection + base extraction
# -----------------------------
SEP_CHARS = r"[_\-\.\|/:#]"
BRACKETED_TOKEN_RE = re.compile(r"(\[[^\]]+\]|\([^\)]+\))")
LONG_DIGITS_RE = re.compile(r"\d{6,}")
MIXED_ID_RE = re.compile(r"(?i)(?:[a-z]*\d+[a-z]+|[a-z]+\d+[a-z]*|\d+[a-z]+\d+)")
DATEISH_RE = re.compile(r"\b(?:19|20)\d{2}[01]\d[0-3]\d(?:[ T]?\d{2}\d{2}\d{2}(?:\d{3,6})?)?\b")
TIMEISH_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d{1,6})?\b")
RESOURCEISH_RE = re.compile(r"(?i)\b(?:system|clerk|manager|worker|nurse|doctor|staff|machine|applicant)[-_]?\d{3,}\b")


def looks_machine_token(tok: str) -> bool:
    t = _norm_space(tok)
    if not t:
        return False
    # Strip surrounding brackets/parentheses for token inspection
    t2 = re.sub(r"^[\[\(]\s*|\s*[\]\)]$", "", t).strip()
    if not t2:
        return False

    # Strong signals
    if LONG_DIGITS_RE.search(t2):
        return True
    if DATEISH_RE.search(t2) or TIMEISH_RE.search(t2):
        return True
    if RESOURCEISH_RE.search(t2):
        return True
    if MIXED_ID_RE.search(t2) and len(t2) >= 6:
        return True

    # Heuristic: mostly non-space alnum with high digit ratio or long random-ish
    alnum = re.sub(r"[^A-Za-z0-9]", "", t2)
    if len(alnum) >= 10:
        digits = sum(ch.isdigit() for ch in alnum)
        if digits / max(1, len(alnum)) >= 0.35:
            return True
        # long mixed-case/letters+digits token
        if re.search(r"[A-Za-z]", alnum) and re.search(r"\d", alnum):
            return True

    return False


def extract_base_and_polluted(activity: str):
    """
    Returns (base_label, is_polluted).
    Base label is derived by removing machine-generated prefix/suffix tokens
    separated by common separators or bracketed tokens.
    """
    raw = _norm_space(activity)
    if not raw:
        return "", False

    s = raw

    # 1) Remove bracketed machine tokens at ends iteratively: [X][ID][TS]
    polluted = False
    # Capture bracketed tokens anywhere; we only remove if they look machine-generated
    # and are at the end or beginning (common pollution pattern).
    # We'll iteratively strip from both ends.
    while True:
        s_strip = s.strip()
        m_end = re.search(r"(\s*(\[[^\]]+\]|\([^\)]+\))\s*)$", s_strip)
        if m_end:
            tok = m_end.group(2)
            if looks_machine_token(tok):
                s = s_strip[: m_end.start()].rstrip()
                polluted = True
                continue
        m_start = re.match(r"^\s*(\[[^\]]+\]|\([^\)]+\))\s*", s_strip)
        if m_start:
            tok = m_start.group(1)
            if looks_machine_token(tok):
                s = s_strip[m_start.end():].lstrip()
                polluted = True
                continue
        break

    # 2) Split by separators and remove machine tokens at ends
    # Keep internal separators as part of base if they don't look machine-generated.
    parts = re.split(SEP_CHARS, s)
    seps = re.findall(SEP_CHARS, s)
    parts = [p for p in parts]  # keep empties for reconstruction logic

    # Trim whitespace in parts
    parts_trim = [_norm_space(p) for p in parts]

    # Identify removable prefix/suffix tokens
    left = 0
    right = len(parts_trim) - 1

    # Remove empty tokens at ends
    while left <= right and parts_trim[left] == "":
        left += 1
    while right >= left and parts_trim[right] == "":
        right -= 1

    # Remove machine-like tokens at ends (prefix/suffix)
    changed = True
    while changed and left <= right:
        changed = False
        if left <= right and looks_machine_token(parts_trim[left]):
            left += 1
            polluted = True
            changed = True
            while left <= right and parts_trim[left] == "":
                left += 1
        if left <= right and looks_machine_token(parts_trim[right]):
            right -= 1
            polluted = True
            changed = True
            while right >= left and parts_trim[right] == "":
                right -= 1

    # Reconstruct base from remaining span using original separators between kept parts.
    if left > right:
        base = _norm_space(raw)  # fallback: cannot infer base safely
        return base, polluted

    # Rebuild using original string slice approach:
    # We'll join kept parts with single spaces to preserve meaningful text.
    kept = parts_trim[left:right + 1]
    base = _norm_space(" ".join([k for k in kept if k != ""]))

    # 3) Special case: resource-like prefix glued to label (e.g., Manager000001_Approve request)
    # If the first token contains a resource-ish pattern and there is more text, drop it.
    if base:
        toks = base.split()
        if toks and RESOURCEISH_RE.fullmatch(toks[0]) and len(toks) > 1:
            base2 = _norm_space(" ".join(toks[1:]))
            if base2:
                polluted = True
                base = base2

    # 4) If base differs materially from raw and removed part looked machine-like, mark polluted.
    # Avoid marking polluted for trivial punctuation-only differences.
    if not polluted:
        # Detect embedded machine token separated by whitespace at end (e.g., "Call advisor (80nSeSg) 20230929-180925")
        # We'll attempt to strip trailing tokens that look machine-generated.
        toks = raw.split()
        if len(toks) >= 2:
            # strip up to 2 trailing tokens if machine-like
            stripped = toks[:]
            removed_any = False
            for _ in range(2):
                if stripped and looks_machine_token(stripped[-1]):
                    stripped = stripped[:-1]
                    removed_any = True
                else:
                    break
            if removed_any:
                base_candidate = _norm_space(" ".join(stripped))
                if base_candidate and base_candidate != raw:
                    polluted = True
                    base = base_candidate

    if not base:
        base = _norm_space(raw)

    return base, polluted


# -----------------------------
# Distorted detection (character-level)
# -----------------------------
def levenshtein_distance(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure a is shorter for memory
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for i, ch_b in enumerate(b, start=1):
        cur = [i]
        for j, ch_a in enumerate(a, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ch_a == ch_b else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    a = _norm_space(a)
    b = _norm_space(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = levenshtein_distance(a.lower(), b.lower())
    denom = max(len(a), len(b))
    return 1.0 - (dist / denom)


def build_canonical_pool(clean_labels, min_freq=2):
    """
    Build canonical candidates from cleaned labels only.
    We keep labels with frequency >= min_freq as initial canonicals,
    then merge near-duplicates by choosing the more frequent as canonical.
    """
    freq = Counter(clean_labels)
    candidates = [lbl for lbl, c in freq.items() if c >= min_freq and lbl]
    # If too few candidates, fall back to top labels (but still cleaned)
    if len(candidates) < 2:
        candidates = [lbl for lbl, _ in freq.most_common(10) if lbl]

    # Merge very-close candidates (typo variants) by frequency preference
    canon = []
    for lbl in sorted(candidates, key=lambda x: (-freq[x], x.lower())):
        placed = False
        for i, c_lbl in enumerate(canon):
            sim = edit_similarity(lbl, c_lbl)
            if sim >= 0.92:
                # keep the more frequent / stable one
                if freq[lbl] > freq[c_lbl]:
                    canon[i] = lbl
                placed = True
                break
        if not placed:
            canon.append(lbl)
    return canon, freq


def detect_distorted(df: pd.DataFrame, base_col: str):
    clean_labels = df[base_col].tolist()
    canon_pool, freq = build_canonical_pool(clean_labels, min_freq=2)

    # For each label, find closest canonical; if very close but not equal -> distorted
    for idx, lbl in enumerate(clean_labels):
        lbl_n = _norm_space(lbl)
        if not lbl_n:
            continue
        best = None
        best_sim = -1.0
        for c in canon_pool:
            if not c:
                continue
            sim = edit_similarity(lbl_n, c)
            if sim > best_sim:
                best_sim = sim
                best = c

        if best is None:
            continue

        if lbl_n != best:
            # Distortion threshold: close at character level, but not identical.
            # Also require small absolute edit distance to avoid semantic differences.
            dist = levenshtein_distance(lbl_n.lower(), best.lower())
            max_len = max(len(lbl_n), len(best))
            # Adaptive constraints
            if best_sim >= 0.88 and dist <= max(2, int(0.15 * max_len)):
                # Prefer treating the less frequent variant as distorted when close
                if freq[lbl_n] <= freq[best]:
                    add_error(df, idx, "distorted")


# -----------------------------
# Synonymous detection (semantic)
# -----------------------------
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def detect_synonymous(df: pd.DataFrame, base_col: str, threshold: float = 0.80):
    labels = df[base_col].astype(str).map(_norm_space).tolist()
    freq = Counter(labels)
    uniq = sorted({l for l in labels if l})
    if len(uniq) < 2:
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(uniq, convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(emb, emb).cpu().numpy()

    uf = UnionFind(len(uniq))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if sim[i, j] >= threshold:
                uf.union(i, j)

    comps = defaultdict(list)
    for i in range(len(uniq)):
        comps[uf.find(i)].append(i)

    # Determine canonical per component (most frequent label in dataset)
    label_to_canon = {}
    for _, idxs in comps.items():
        if len(idxs) < 2:
            continue
        members = [uniq[i] for i in idxs]
        canonical = sorted(members, key=lambda x: (-freq[x], x.lower()))[0]
        for m in members:
            label_to_canon[m] = canonical

    # Flag non-canonical members
    for idx, lbl in enumerate(labels):
        if not lbl:
            continue
        canon = label_to_canon.get(lbl)
        if canon and canon != lbl:
            add_error(df, idx, "synonymous")


# -----------------------------
# Form-based detection (timestamp collisions within case)
# -----------------------------
def detect_form_based(df: pd.DataFrame):
    # Within each case, identical timestamps with >=2 events and different activities
    # Mark all events in such clusters.
    for case_id, g in df.groupby("Case", sort=False):
        # group by exact timestamp string to avoid parsing artifacts; still parse elsewhere for collateral
        for ts, gg in g.groupby("Timestamp", sort=False):
            if len(gg) >= 2:
                acts = gg["Activity"].astype(str).map(_norm_space).tolist()
                if len(set(acts)) >= 2:
                    for idx in gg.index:
                        add_error(df, idx, "form-based")


# -----------------------------
# Collateral detection (bursts within case)
# -----------------------------
def detect_collateral(df: pd.DataFrame):
    # Parse timestamps
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_ts_parsed"] = ts

    for case_id, g in df.groupby("Case", sort=False):
        g2 = g.sort_values(["_ts_parsed", "Timestamp"], kind="mergesort")
        idxs = g2.index.to_list()
        times = g2["_ts_parsed"].to_list()
        resources = g2["Resource"].astype(str).map(_norm_space).to_list()

        # Identify consecutive bursts where delta<=5s or identical timestamps, and same resource
        burst = [idxs[0]] if idxs else []
        for k in range(1, len(idxs)):
            t_prev, t_cur = times[k - 1], times[k]
            r_prev, r_cur = resources[k - 1], resources[k]
            close = False
            if pd.isna(t_prev) or pd.isna(t_cur):
                # fallback: if raw timestamp strings identical, treat as close
                close = _norm_space(df.at[idxs[k - 1], "Timestamp"]) == _norm_space(df.at[idxs[k], "Timestamp"])
            else:
                delta = (t_cur - t_prev).total_seconds()
                close = (delta <= 5.0) or (delta == 0.0)

            if close and r_prev and r_cur and (r_prev == r_cur):
                burst.append(idxs[k])
            else:
                if len(burst) >= 2:
                    for bi in burst:
                        add_error(df, bi, "collateral")
                burst = [idxs[k]]

        if len(burst) >= 2:
            for bi in burst:
                add_error(df, bi, "collateral")

    df.drop(columns=["_ts_parsed"], inplace=True)


# -----------------------------
# Homonymous detection (same label, different preceding activity within case)
# -----------------------------
def detect_homonymous(df: pd.DataFrame, base_col: str):
    # Use cleaned base label for "same activity label" comparisons to avoid pollution artifacts.
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_ts_parsed"] = ts

    for case_id, g in df.groupby("Case", sort=False):
        g2 = g.sort_values(["_ts_parsed", "Timestamp"], kind="mergesort")
        idxs = g2.index.to_list()
        labels = g2[base_col].astype(str).map(_norm_space).to_list()

        # Map label -> list of (pos, idx, preceding_label)
        occ = defaultdict(list)
        for pos, (idx, lbl) in enumerate(zip(idxs, labels)):
            if not lbl:
                continue
            prev_lbl = labels[pos - 1] if pos - 1 >= 0 else ""
            occ[lbl].append((pos, idx, prev_lbl))

        for lbl, items in occ.items():
            if len(items) < 2:
                continue

            # Prefer collateral if occurrences are 3+ (per instruction)
            if len(items) >= 3:
                continue

            # Exactly 2 occurrences: check clearly different preceding context
            (pos1, idx1, prev1), (pos2, idx2, prev2) = items[0], items[1]
            prev1n, prev2n = _norm_space(prev1), _norm_space(prev2)
            if prev1n and prev2n and prev1n.lower() != prev2n.lower():
                # Also ensure not trivially adjacent burst (which would be collateral-like)
                t1 = df.at[idx1, "_ts_parsed"]
                t2 = df.at[idx2, "_ts_parsed"]
                if not (pd.notna(t1) and pd.notna(t2) and abs((t2 - t1).total_seconds()) <= 5.0):
                    add_error(df, idx1, "homonymous")
                    add_error(df, idx2, "homonymous")

    df.drop(columns=["_ts_parsed"], inplace=True)


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python script.py input.csv output.csv")

    inp = sys.argv[1]
    outp = sys.argv[2]

    df = pd.read_csv(inp)

    required = ["Case", "Activity", "Timestamp", "Resource"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Initialize required output columns (exact names) and internal error set
    df["error_detected"] = False
    df["detected_error_types"] = ""
    df["_err_set"] = [set() for _ in range(len(df))]

    # Normalize key columns (do not create extra output columns)
    df["Case"] = df["Case"].apply(_norm_space)
    df["Activity"] = df["Activity"].apply(_norm_space)
    df["Timestamp"] = df["Timestamp"].apply(_norm_space)
    df["Resource"] = df["Resource"].apply(_norm_space)

    # --- Polluted + base label extraction (independent layer) ---
    base_labels = []
    polluted_flags = []
    for i, act in enumerate(df["Activity"].tolist()):
        base, is_polluted = extract_base_and_polluted(act)
        base_labels.append(base)
        polluted_flags.append(is_polluted)
        if is_polluted:
            add_error(df, i, "polluted")

    df["_base_activity"] = base_labels  # internal only
    # --- Form-based (temporal collisions) ---
    detect_form_based(df)

    # --- Collateral (bursts) ---
    detect_collateral(df)

    # --- Distorted (character corruption on base label only) ---
    detect_distorted(df, "_base_activity")

    # --- Synonymous (semantic similarity on base label only) ---
    detect_synonymous(df, "_base_activity", threshold=0.80)

    # --- Homonymous (contextual path differences within case; prefer collateral if 3+) ---
    detect_homonymous(df, "_base_activity")

    # Finalize output columns
    finalize_errors(df)

    # Drop internal helper column(s)
    df.drop(columns=["_base_activity"], inplace=True)

    # Save
    df.to_csv(outp, index=False)


if __name__ == "__main__":
    main()