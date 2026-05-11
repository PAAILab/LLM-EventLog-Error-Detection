import sys
import re
from collections import Counter, defaultdict
import pandas as pd

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Utilities
# -----------------------------
VALID_ERROR_TYPES = ["form-based", "polluted", "distorted", "synonymous", "collateral", "homonymous"]


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["Case", "Activity", "Timestamp", "Resource"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Robust parsing; keep NaT for unparseable timestamps
    return pd.to_datetime(series, errors="coerce")


def _normalize_spaces(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _lower_alnum_signature(s: str) -> str:
    # For some heuristics; keep letters/digits/spaces
    s = _normalize_spaces(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _add_error(df: pd.DataFrame, idxs, err_type: str) -> None:
    if err_type not in VALID_ERROR_TYPES:
        raise ValueError(f"Invalid error type: {err_type}")
    if len(idxs) == 0:
        return
    df.loc[idxs, "error_detected"] = True
    # Append pipe-separated without duplicates
    current = df.loc[idxs, "detected_error_types"].fillna("")
    def merge_one(x):
        if not x:
            return err_type
        parts = [p for p in x.split("|") if p]
        if err_type not in parts:
            parts.append(err_type)
        return "|".join(parts)
    df.loc[idxs, "detected_error_types"] = current.map(merge_one)


# -----------------------------
# Polluted detection + base label extraction
# -----------------------------
_POLLUTION_SEP = r"[_\-\.\|:/#]"
# Common machine-ish tokens: long digits, timestamp fragments, mixed-case IDs, resource-like prefixes
_RE_LONG_DIGITS = re.compile(r"\d{6,}")
_RE_TS_FRAGMENT = re.compile(r"(19|20)\d{2}[01]\d[0-3]\d([ T]?[0-2]\d[0-5]\d[0-5]\d(\d{3,6})?)?")
_RE_MIXED_ID = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{6,}")
_RE_RESOURCEISH = re.compile(r"(?i)\b(?:system|clerk|manager|worker|nurse|doctor|staff|machine|applicant|logistics|lab)[-_]?\d{3,}\b")
_RE_BRACKETED = re.compile(r"[\[\(\{].{2,}[\]\)\}]")

def _token_is_machiney(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    # Very long token
    if len(t) >= 12 and re.search(r"[A-Za-z0-9]", t):
        # long alnum often machiney
        if _RE_LONG_DIGITS.search(t) or _RE_MIXED_ID.search(t) or _RE_TS_FRAGMENT.search(t):
            return True
    # Long digits / timestamp fragments
    if _RE_LONG_DIGITS.search(t) or _RE_TS_FRAGMENT.search(t):
        return True
    # Mixed alnum id
    if _RE_MIXED_ID.search(t):
        return True
    # Resource-like prefix
    if _RE_RESOURCEISH.search(t):
        return True
    # Bracketed payloads often machiney when combined
    if _RE_BRACKETED.search(t) and (re.search(r"\d", t) or _RE_MIXED_ID.search(t)):
        return True
    return False


def extract_base_and_polluted(activity: str):
    """
    Returns: (base_label:str, is_polluted:bool)
    Strategy:
      - Split on common separators and also consider bracketed segments.
      - Remove leading/trailing machiney tokens while preserving meaningful base text.
      - If removal changes the label and removed part is machiney => polluted.
    """
    raw = "" if activity is None else str(activity)
    raw_norm = _normalize_spaces(raw)

    if not raw_norm:
        return raw_norm, False

    # Special case: bracketed concatenations like [Cook Dinner][BxA81Qe][214503882]
    # Extract bracket contents; if first bracket looks human and later brackets machiney -> polluted
    bracket_contents = re.findall(r"\[([^\]]+)\]", raw_norm)
    if len(bracket_contents) >= 2:
        first = _normalize_spaces(bracket_contents[0])
        rest = bracket_contents[1:]
        if first and any(_token_is_machiney(_normalize_spaces(x)) for x in rest):
            return first, True

    # Split on separators but keep original spacing in base
    parts = re.split(_POLLUTION_SEP, raw_norm)
    parts = [_normalize_spaces(p) for p in parts if _normalize_spaces(p)]

    # If no separators, still might be polluted by appended id without separator (e.g., CookDinner74lwMHm2023...)
    # Heuristic: if contains a long mixed id or long digits and also contains letters/spaces -> try to strip tail
    if len(parts) == 1:
        s = parts[0]
        # Try to split tail machiney chunk
        m = re.search(r"(.+?)(\s*(?:\d{6,}|(19|20)\d{2}[01]\d[0-3]\d.*|[A-Za-z0-9]{8,}))$", s)
        if m:
            head = _normalize_spaces(m.group(1))
            tail = _normalize_spaces(m.group(2))
            if head and _token_is_machiney(tail):
                return head, True
        # Try camel-case / missing space base like CookDinner + id
        m2 = re.search(r"^([A-Za-z]+(?:\s*[A-Za-z]+)+)([A-Za-z0-9]{8,})$", s)
        if m2:
            head = _normalize_spaces(m2.group(1))
            tail = _normalize_spaces(m2.group(2))
            if head and _token_is_machiney(tail):
                # Also normalize head by inserting space between lower->upper transitions
                head2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", head)
                head2 = _normalize_spaces(head2)
                return head2, True
        return raw_norm, False

    # Remove machiney tokens from ends; keep middle as base
    left = 0
    right = len(parts) - 1
    removed_any = False

    while left <= right and _token_is_machiney(parts[left]):
        removed_any = True
        left += 1
    while right >= left and _token_is_machiney(parts[right]):
        removed_any = True
        right -= 1

    base_parts = parts[left:right + 1] if left <= right else []
    base = _normalize_spaces(" ".join(base_parts))

    # If we removed something machiney and base is non-empty and differs materially -> polluted
    if removed_any and base and _lower_alnum_signature(base) != _lower_alnum_signature(raw_norm):
        return base, True

    # Another heuristic: prefix like Manager000001_Approve request
    # If first part looks resourceish and remaining looks human -> polluted
    if len(parts) >= 2 and _RE_RESOURCEISH.search(parts[0]) and len(_normalize_spaces(" ".join(parts[1:]))) >= 3:
        base2 = _normalize_spaces(" ".join(parts[1:]))
        if base2:
            return base2, True

    return raw_norm, False


# -----------------------------
# Distorted detection (edit similarity)
# -----------------------------
def _levenshtein_distance(a: str, b: str) -> int:
    # Iterative DP, O(len(a)*len(b)) with rolling row
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    a_len, b_len = len(a), len(b)
    if a_len > b_len:
        a, b = b, a
        a_len, b_len = b_len, a_len
    prev = list(range(a_len + 1))
    for j in range(1, b_len + 1):
        bj = b[j - 1]
        cur = [j] + [0] * a_len
        for i in range(1, a_len + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(
                prev[i] + 1,      # deletion
                cur[i - 1] + 1,   # insertion
                prev[i - 1] + cost  # substitution
            )
        prev = cur
    return prev[a_len]


def _normalized_edit_similarity(a: str, b: str) -> float:
    a = _normalize_spaces(a).lower()
    b = _normalize_spaces(b).lower()
    if not a and not b:
        return 1.0
    dist = _levenshtein_distance(a, b)
    denom = max(len(a), len(b), 1)
    return 1.0 - (dist / denom)


def detect_distorted(df: pd.DataFrame, base_col: str):
    """
    Build canonical candidates from cleaned base labels only.
    Prefer more frequent label as canonical when two labels are very close.
    Flag less frequent close variants as distorted.
    """
    base_labels = df[base_col].fillna("").map(_normalize_spaces)
    counts = Counter(base_labels.tolist())

    unique = [lbl for lbl in counts.keys() if lbl]
    if len(unique) <= 1:
        return set(), {}  # no distorted

    # Sort by frequency desc, then length desc (stability heuristic)
    sorted_labels = sorted(unique, key=lambda x: (counts[x], len(x)), reverse=True)

    # Determine canonical set by absorbing close variants into a canonical representative
    # Thresholds tuned for typos: high similarity and small absolute distance
    SIM_THR = 0.88
    MAX_ABS_DIST = 3

    canonical = []
    mapping_variant_to_canon = {}

    for lbl in sorted_labels:
        if lbl in mapping_variant_to_canon:
            continue
        # Try to map to an existing canonical if very close and canonical is more frequent/stable
        mapped = False
        for canon in canonical:
            sim = _normalized_edit_similarity(lbl, canon)
            dist = _levenshtein_distance(_normalize_spaces(lbl).lower(), _normalize_spaces(canon).lower())
            if sim >= SIM_THR and dist <= MAX_ABS_DIST:
                # lbl is a variant of canon (canon already chosen earlier due to sorting)
                mapping_variant_to_canon[lbl] = canon
                mapped = True
                break
        if not mapped:
            canonical.append(lbl)

    # Second pass: map remaining labels to best canonical if close enough
    for lbl in sorted_labels:
        if lbl in mapping_variant_to_canon:
            continue
        best = None
        best_sim = -1.0
        best_dist = 10**9
        for canon in canonical:
            if canon == lbl:
                continue
            sim = _normalized_edit_similarity(lbl, canon)
            dist = _levenshtein_distance(_normalize_spaces(lbl).lower(), _normalize_spaces(canon).lower())
            if sim > best_sim or (sim == best_sim and dist < best_dist):
                best_sim, best_dist, best = sim, dist, canon
        if best is not None and best_sim >= SIM_THR and best_dist <= MAX_ABS_DIST:
            # Only treat as distorted if canonical is at least as frequent as variant
            if counts[best] >= counts[lbl]:
                mapping_variant_to_canon[lbl] = best

    distorted_idxs = set()
    distorted_target = {}  # idx -> canonical label
    for idx, lbl in base_labels.items():
        if not lbl:
            continue
        canon = mapping_variant_to_canon.get(lbl)
        if canon and canon != lbl:
            distorted_idxs.add(idx)
            distorted_target[idx] = canon

    return distorted_idxs, distorted_target


# -----------------------------
# Form-based detection
# -----------------------------
def detect_formbased(df: pd.DataFrame, ts_col: str, case_col: str, act_col: str):
    """
    Within each case, find timestamp clusters with >=2 events with identical timestamps
    and different activity labels. Mark all events in such clusters.
    """
    form_idxs = set()
    # Use raw activity (not base) because definition is about different activity labels recorded at same timestamp
    # but still within same case.
    for case_id, g in df.groupby(case_col, sort=False):
        # Only consider parseable timestamps; NaT cannot form identical timestamp clusters reliably
        gg = g.dropna(subset=[ts_col])
        if gg.empty:
            continue
        # group by exact timestamp value
        for ts, h in gg.groupby(ts_col, sort=False):
            if len(h) < 2:
                continue
            acts = h[act_col].fillna("").map(_normalize_spaces)
            if acts.nunique(dropna=False) >= 2:
                form_idxs.update(h.index.tolist())
    return form_idxs


# -----------------------------
# Collateral detection
# -----------------------------
def detect_collateral(df: pd.DataFrame, ts_col: str, case_col: str, res_col: str):
    """
    Within each case, find consecutive bursts where:
      - consecutive events within 5 seconds OR identical timestamps
      - and share strong contextual similarity such as same resource
    Flag all events in bursts of size >=2.
    """
    collateral_idxs = set()
    WINDOW_SEC = 5

    for case_id, g in df.groupby(case_col, sort=False):
        gg = g.copy()
        gg = gg.sort_values(ts_col, kind="mergesort")  # stable
        # If timestamps missing, cannot compute gaps reliably; skip those rows for burst logic
        # but keep them in sequence? We'll only evaluate adjacent rows with valid timestamps.
        idxs = gg.index.tolist()
        times = gg[ts_col].tolist()
        resources = gg[res_col].fillna("").map(_normalize_spaces).tolist()

        # Build bursts on consecutive events with valid timestamps
        current_burst = [idxs[0]] if idxs else []
        for k in range(1, len(idxs)):
            prev_i = idxs[k - 1]
            cur_i = idxs[k]
            t_prev = times[k - 1]
            t_cur = times[k]
            r_prev = resources[k - 1]
            r_cur = resources[k]

            close_in_time = False
            same_resource = (r_prev != "" and r_prev == r_cur)

            if pd.notna(t_prev) and pd.notna(t_cur):
                delta = abs((t_cur - t_prev).total_seconds())
                if delta <= WINDOW_SEC or t_cur == t_prev:
                    close_in_time = True

            if close_in_time and same_resource:
                current_burst.append(cur_i)
            else:
                if len(current_burst) >= 2:
                    collateral_idxs.update(current_burst)
                current_burst = [cur_i]

        if len(current_burst) >= 2:
            collateral_idxs.update(current_burst)

    return collateral_idxs


# -----------------------------
# Synonymous detection (sentence-transformers)
# -----------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

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


def detect_synonymous(df: pd.DataFrame, base_col: str):
    """
    Embed unique cleaned labels (base_col) and cluster by cosine similarity >= 0.80.
    Canonical per cluster = most frequent label in dataset.
    Flag rows whose base label is a non-canonical cluster member.
    """
    base = df[base_col].fillna("").map(_normalize_spaces)
    counts = Counter(base.tolist())
    unique_labels = [l for l in counts.keys() if l]
    if len(unique_labels) < 2:
        return set(), {}  # no synonyms

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_labels, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    thr = 0.80
    n = len(unique_labels)
    uf = UnionFind(n)

    # Pairwise union above threshold (excluding i==j)
    # For reproducibility, deterministic iteration order
    for i in range(n):
        # Only check j>i
        row = sim_matrix[i]
        for j in range(i + 1, n):
            if float(row[j]) >= thr:
                uf.union(i, j)

    # Build clusters
    clusters = defaultdict(list)
    for i, lbl in enumerate(unique_labels):
        clusters[uf.find(i)].append(lbl)

    # Determine canonical per cluster and mapping
    label_to_canon = {}
    for _, members in clusters.items():
        if len(members) < 2:
            continue
        # canonical = most frequent; tie-breaker: longer label then lexicographic
        canon = sorted(members, key=lambda x: (counts[x], len(x), x), reverse=True)[0]
        for m in members:
            label_to_canon[m] = canon

    syn_idxs = set()
    syn_target = {}
    for idx, lbl in base.items():
        if not lbl:
            continue
        canon = label_to_canon.get(lbl)
        if canon and canon != lbl:
            syn_idxs.add(idx)
            syn_target[idx] = canon

    return syn_idxs, syn_target


# -----------------------------
# Homonymous detection
# -----------------------------
def detect_homonymous(df: pd.DataFrame, case_col: str, base_col: str, ts_col: str):
    """
    Within each case, for labels that appear more than once:
      - examine immediately preceding activity (base label) for each occurrence
      - if exactly 2 occurrences and preceding contexts differ -> flag both as homonymous
      - if 3+ occurrences, prefer collateral (do not flag homonymous here)
    """
    hom_idxs = set()

    for case_id, g in df.groupby(case_col, sort=False):
        gg = g.sort_values(ts_col, kind="mergesort")
        base_seq = gg[base_col].fillna("").map(_normalize_spaces).tolist()
        idx_seq = gg.index.tolist()

        # positions per label
        pos_by_label = defaultdict(list)
        for pos, lbl in enumerate(base_seq):
            if lbl:
                pos_by_label[lbl].append(pos)

        for lbl, positions in pos_by_label.items():
            if len(positions) < 2:
                continue
            if len(positions) >= 3:
                # Prefer collateral over homonymous when 3+ occurrences
                continue

            # Exactly 2 occurrences
            p1, p2 = positions[0], positions[1]
            prev1 = base_seq[p1 - 1] if p1 - 1 >= 0 else ""
            prev2 = base_seq[p2 - 1] if p2 - 1 >= 0 else ""

            # "consistently different across occurrences" for 2 occurrences => prev differs and both non-empty
            if prev1 and prev2 and prev1 != prev2:
                hom_idxs.add(idx_seq[p1])
                hom_idxs.add(idx_seq[p2])

    return hom_idxs


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python script.py input.csv output.csv")

    inp = sys.argv[1]
    out = sys.argv[2]

    df = pd.read_csv(inp)
    _ensure_required_columns(df)

    # Initialize required output columns
    df["error_detected"] = False
    df["detected_error_types"] = ""

    # Parse timestamps (do not overwrite original column; but we need datetime for temporal logic)
    df["_ts"] = _safe_to_datetime(df["Timestamp"])

    # Normalize key text fields (do not overwrite originals)
    df["_activity_raw_norm"] = df["Activity"].fillna("").map(_normalize_spaces)
    df["_resource_norm"] = df["Resource"].fillna("").map(_normalize_spaces)

    # Polluted + base extraction
    base_labels = []
    polluted_flags = []
    for a in df["_activity_raw_norm"].tolist():
        base, is_polluted = extract_base_and_polluted(a)
        base_labels.append(base)
        polluted_flags.append(bool(is_polluted))
    df["_activity_base"] = pd.Series(base_labels, index=df.index).map(_normalize_spaces)
    df["_is_polluted"] = polluted_flags

    polluted_idxs = df.index[df["_is_polluted"]].tolist()
    _add_error(df, polluted_idxs, "polluted")

    # Form-based
    form_idxs = detect_formbased(df, ts_col="_ts", case_col="Case", act_col="_activity_raw_norm")
    _add_error(df, list(form_idxs), "form-based")

    # Collateral
    collateral_idxs = detect_collateral(df, ts_col="_ts", case_col="Case", res_col="_resource_norm")
    _add_error(df, list(collateral_idxs), "collateral")

    # Distorted (on base label only, after pollution removal)
    distorted_idxs, _ = detect_distorted(df, base_col="_activity_base")
    _add_error(df, list(distorted_idxs), "distorted")

    # Synonymous (on base label only, after pollution removal)
    syn_idxs, _ = detect_synonymous(df, base_col="_activity_base")
    _add_error(df, list(syn_idxs), "synonymous")

    # Homonymous (on base label; prefer collateral when 3+ occurrences)
    hom_idxs = detect_homonymous(df, case_col="Case", base_col="_activity_base", ts_col="_ts")
    _add_error(df, list(hom_idxs), "homonymous")

    # Ensure detected_error_types is empty string for clean rows
    df.loc[~df["error_detected"], "detected_error_types"] = ""

    # Drop internal columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # Save
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()