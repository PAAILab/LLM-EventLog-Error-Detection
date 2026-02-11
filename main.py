from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

try:
    from openai import RateLimitError, APIStatusError
except Exception:
    RateLimitError = Exception
    APIStatusError = Exception


REQUIRED_COLUMNS = [
    "error_detected",
    "detected_error_types",
]


def call_llm_with_logging(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    """Calls chat.completions"""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content

    return text


def parse_ground_truth_label(label_str) -> set:
    """Parse ground truth label to extract error types"""
    if pd.isna(label_str) or str(label_str).strip() == "":
        return set()

    label_lower = str(label_str).lower()
    error_types = set()

    if "autoformbased" in label_lower or "form-based" in label_lower or "form_based" in label_lower or "formbased" in label_lower:
        error_types.add("form-based")
    if "polluted" in label_lower:
        error_types.add("polluted")
    if "distorted" in label_lower:
        error_types.add("distorted")
    if "synonymous" in label_lower:
        error_types.add("synonymous")
    if "collateral" in label_lower:
        error_types.add("collateral")
    if "homonymous" in label_lower:
        error_types.add("homonymous")
    if "empty" in label_lower:
        error_types.add("empty")

    return error_types


def compute_metrics_from_output(output_csv: Path, ground_truth_csv: Path) -> Dict[str, Any]:
    """
    Compute metrics by comparing LLM output with ground truth in THREE styles:
    
    1. STRICT: Exact match - all GT error types must exactly match all detected types
    2. MODERATE: Exact subset match - detected types must exactly match GT (no more, no less)
    3. GENEROUS: Any overlap - at least one error type matches
    
    Ground truth CSV should have a 'label' or 'error_type' column with error types.
    LLM output CSV should have 'error_detected' and 'detected_error_types' columns.
    """
    # Load ground truth
    df_gt = pd.read_csv(ground_truth_csv)
    
    # Find the label column (try multiple common names)
    label_col = None
    for col_name in ['label', 'error_type', 'error_types', 'gt_error_type', 'is_error']:
        if col_name in df_gt.columns:
            label_col = col_name
            break
    
    if label_col is None:
        return {
            "error": "Ground truth CSV missing label column (tried: label, error_type, error_types, gt_error_type, is_error)",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": None
        }
    
    # Load LLM output
    df_out = pd.read_csv(output_csv)
    if 'error_detected' not in df_out.columns or 'detected_error_types' not in df_out.columns:
        return {
            "error": "Output CSV missing required columns (need: error_detected, detected_error_types)",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": None
        }
    
    # Ensure both dataframes have same length and alignment
    if len(df_gt) != len(df_out):
        return {
            "error": f"Row count mismatch: GT={len(df_gt)}, Output={len(df_out)}",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": None
        }
    
    def normalize_error_type(t: str) -> str:
        if not t:
            return ""
        t = t.lower().strip()
        t = t.replace("_", "-")
        t = t.replace("formbased", "form-based")
        return t

    # Parse ground truth labels
    df_gt['gt_error_types'] = df_gt[label_col].apply(lambda x: {normalize_error_type(t) for t in parse_ground_truth_label(x)})
    df_gt['gt_has_error'] = df_gt['gt_error_types'].apply(lambda x: len(x) > 0)
    
    # Get predictions
    y_true = df_gt['gt_has_error']
    y_pred = df_out['error_detected'].fillna(False).astype(bool)
    
    CANONICAL_MAP = {
        "form_based": "form-based",
        "formbased": "form-based",
        "form-based": "form-based",
        "autoformbased": "form-based", 
        "distorted": "distorted",
        "polluted": "polluted", 
        "homonymous": "homonymous",
        "synonymous": "synonymous",
        "collateral": "collateral", 
        "empty": "empty"
    }
    
    def canonicalize(t: str) -> str:
        if not t or pd.isna(t):
            return ""
        t = t.lower().strip().replace("_", "-")
        return CANONICAL_MAP.get(t, t)
    
    # Extract predicted error types for each row
    def get_predicted_types(idx):
        pred_types = set()
        all_types = str(df_out.iloc[idx].get("detected_error_types", ""))
        for t in all_types.split("|"):
            canonical = canonicalize(t)
            if canonical:
                pred_types.add(canonical)
        return pred_types
    
    # STYLE 1: STRICT - Exact match (all GT types must match all detected types)
    def is_correct_detection_strict(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # Exact match: sets must be identical
        return gt_types == pred_types
    
    # STYLE 2: MODERATE - Must predict all GT types and no extra types
    def is_correct_detection_moderate(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # Must contain all GT types and nothing extra
        return pred_types.issubset(gt_types)
    
    # STYLE 3: GENEROUS - At least one match
    def is_correct_detection_generous(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # Any overlap
        return len(gt_types & pred_types) > 0
    
    # Calculate metrics for each style
    results = {}
    
    for style_name, is_correct_fn in [
        ("strict", is_correct_detection_strict),
        ("moderate", is_correct_detection_moderate),
        ("generous", is_correct_detection_generous)
    ]:
        TP = sum(1 for i in range(len(df_gt)) if y_true.iloc[i] and y_pred.iloc[i] and is_correct_fn(i))
        FP = sum(1 for i in range(len(df_gt)) if not y_true.iloc[i] and y_pred.iloc[i])
        FN = sum(1 for i in range(len(df_gt)) if y_true.iloc[i] and not (y_pred.iloc[i] and is_correct_fn(i)))
        TN = sum(1 for i in range(len(df_gt)) if not y_true.iloc[i] and not y_pred.iloc[i])
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / len(df_gt) if len(df_gt) > 0 else 0.0
        
        results[f"metrics_{style_name}"] = {
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy
            },
            "confusion_matrix": {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN
            },
            "counts": {
                "total_rows": len(df_gt),
                "ground_truth_errors": int(y_true.sum()),
                "predicted_errors": int(y_pred.sum())
            }
        }
    
    return results


def slugify(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)


def find_repo_root(start_path: Path) -> Path:
    current = start_path
    while current != current.parent:
        if (current / ".git").exists() or (current / "prompts").exists():
            return current
        current = current.parent
    return start_path.parent


def load_prompt_text(repo_root: Path, rel_path: str, default: str = "") -> str:
    """Load prompt from file if exists, otherwise return default"""
    p = repo_root / rel_path
    if p.exists():
        return p.read_text(encoding="utf-8")
    return default


@dataclass
class DetectionContext:
    dataset_stem: str
    input_csv: Path
    ground_truth_csv: Optional[Path]
    output_csv: Path
    results_json_path: Path
    script_path: Path

    detection_summary: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# DetectorAgent (Single Run - No Iteration)
# -----------------------------
class DetectorAgent:
    """
    Detector agent that runs ONCE without iteration/feedback loop.
    """

    def __init__(self, client: OpenAI, model: str, prompt_template: str, scripts_dir: Path):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.scripts_dir = scripts_dir

    def execute(self, context: DetectionContext) -> DetectionContext:
        print("\n[DETECTOR] Running single detection pass...")

        # Load input data
        df = pd.read_csv(context.input_csv)
        sample = df.head(100).to_csv(index=False)

        # Build user message with all possible template variables
        template_vars = {
            "dataset_name": context.input_csv.name,
            "dataset_sample": sample,
            "INPUT_CSV_PATH": str(context.input_csv),
            "OUTPUT_CSV_PATH": str(context.output_csv),
            "input_csv": str(context.input_csv),
            "output_csv": str(context.output_csv),
        }
        
        # Use a safer formatting approach that ignores missing keys
        user_msg = self.prompt_template
        for key, value in template_vars.items():
            user_msg = user_msg.replace(f"{{{key}}}", str(value))

        max_retries = 3
        for attempt in range(max_retries):
            print(f"[DETECTOR] Attempt {attempt + 1}/{max_retries}")
            
            # Call LLM (first time with original prompt, subsequent times with error feedback)
            if attempt == 0:
                response_text = call_llm_with_logging(
                    client=self.client,
                    model=self.model,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.1,
                )
            else:
                # Retry with error feedback
                retry_prompt = f"""The previous script failed with this error:

ERROR:
{last_error}

FAILED SCRIPT (first 2000 chars):
{last_script[:2000]}

Please fix the script to resolve this error. Remember:
- Use sys.argv[1] for input path and sys.argv[2] for output path
- Do NOT hardcode file paths
- Use pandas 2.0+ compatible code
- Add exactly TWO columns: error_detected (boolean) and detected_error_types (string)
- Keep ALL original columns from the input CSV

Return ONLY the corrected Python code wrapped in ```python``` markers."""

                response_text = call_llm_with_logging(
                    client=self.client,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"```python\n{last_script}\n```"},
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.1,
                )

            # Parse response
            parsed = self._parse_detection_response(response_text)
            
            # Generate detection script
            script_content = parsed.get("python_code", "")
            if not script_content:
                raise RuntimeError("LLM did not return python_code")
            
            # Fix common pandas compatibility issues
            script_content = self._fix_pandas_compatibility(script_content)

            context.script_path.parent.mkdir(parents=True, exist_ok=True)
            context.script_path.write_text(script_content, encoding="utf-8")

            # Execute detection script
            print(f"[DETECTOR] Executing detection script: {context.script_path}")
            print(f"[DETECTOR] Input: {context.input_csv}")
            print(f"[DETECTOR] Output: {context.output_csv}")
            
            result = subprocess.run(
                [sys.executable, str(context.script_path), str(context.input_csv), str(context.output_csv)],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Success! Break out of retry loop
                print(f"[DETECTOR] Script executed successfully on attempt {attempt + 1}")
                break
            else:
                # Script failed - prepare for retry
                last_error = f"STDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
                last_script = script_content
                
                print(f"\n[DETECTOR] Script failed on attempt {attempt + 1}")
                print(f"[DETECTOR] Error preview: {result.stderr[:200]}...")
                
                if attempt == max_retries - 1:
                    # This was the last attempt - give up
                    error_msg = f"Detection script failed after {max_retries} attempts:\n{last_error}"
                    print(f"\n[ERROR] {error_msg}")
                    print(f"\n[DEBUG] Final script content:")
                    print("=" * 60)
                    print(script_content[:1000])
                    print("=" * 60)
                    raise RuntimeError(error_msg)
                else:
                    print(f"[DETECTOR] Retrying with error feedback...")

        # Validate output
        if not context.output_csv.exists():
            raise RuntimeError(f"Detection script did not create output: {context.output_csv}")

        df_out = pd.read_csv(context.output_csv)
        missing = [c for c in REQUIRED_COLUMNS if c not in df_out.columns]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")

        errors_detected = int(df_out["error_detected"].sum())
        context.detection_summary = {
            "total_rows": len(df_out),
            "errors_detected": errors_detected,
        }

        print(f"[DETECTOR] Detection complete - {errors_detected} errors found in {len(df_out)} rows")

        return context

    def _parse_detection_response(self, text: str) -> Dict[str, Any]:
        """Extract python_code from LLM response"""
        # Try to find code block
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return {"python_code": code_match.group(1)}
        
        # If no code block found, return the whole text
        return {"python_code": text}
    
    def _fix_pandas_compatibility(self, code: str) -> str:
        """Fix common pandas compatibility issues in generated code"""
        # Remove deprecated infer_datetime_format parameter
        code = re.sub(
            r'pd\.to_datetime\([^)]*infer_datetime_format\s*=\s*True[^)]*\)',
            lambda m: m.group(0).replace('infer_datetime_format=True', '').replace(', ,', ',').replace('(,', '(').replace(',)', ')'),
            code
        )
        
        # More robust fix: remove the parameter and clean up commas
        code = re.sub(r',\s*infer_datetime_format\s*=\s*(?:True|False)', '', code)
        code = re.sub(r'infer_datetime_format\s*=\s*(?:True|False)\s*,\s*', '', code)
        
        # Remove utc=False parameter (it's the default)
        code = re.sub(r',\s*utc\s*=\s*False', '', code)
        code = re.sub(r'utc\s*=\s*False\s*,\s*', '', code)
        
        return code


# -----------------------------
# DetectionPipeline
# -----------------------------
class DetectionPipeline:
    """
    Orchestrates only detector and summarizer agents.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        repo_root: Path,
        detector_prompt: str,
    ):
        scripts_dir = repo_root / "script"  # Changed from "scripts" to "script"
        self.detector = DetectorAgent(client, model, detector_prompt, scripts_dir)
        self.model = model


    @staticmethod
    def find_ground_truth(input_csv: Path) -> Optional[Path]:
        """
        Find ground truth file using three possible patterns:
        1. Same file with error_type column
        2. Separate file with _labels, _label, _gt, or _groundtruth suffix
        3. File with _withLabel suffix (replacing _noLabel)
        """
        # Check if input file itself has error_type column
        try:
            df = pd.read_csv(input_csv, nrows=1)
            for col in ['label', 'error_type', 'error_types', 'gt_error_type', 'is_error']:
                if col in df.columns:
                    return input_csv  # Ground truth is in the same file
        except Exception:
            pass
        
        # Look for separate label files
        base_name = input_csv.stem
        parent_dir = input_csv.parent
        
        # Try different label file patterns
        for suffix in ['_labels', '_label', '_gt', '_groundtruth']:
            label_path = parent_dir / f"{base_name}{suffix}.csv"
            if label_path.exists():
                return label_path
        
        # Try _withLabel pattern (for _noLabel files)
        if "_noLabel" in base_name:
            gt_name = base_name.replace("_noLabel", "_withLabel")
            gt_path = parent_dir / f"{gt_name}.csv"
            if gt_path.exists():
                return gt_path
        
        return None  # No ground truth found



    def process_dataset(self, input_csv: Path, detected_output_dir: Path, results_dir: Path) -> DetectionContext:
        dataset_stem = slugify(input_csv.stem)
        
        # Find ground truth (may be None if no labels exist)
        ground_truth_csv = self.find_ground_truth(input_csv)
        
        # Output files according to README structure
        output_csv = detected_output_dir / f"{input_csv.stem}_detected.csv"
        results_json_path = results_dir / f"{self.model}_detection_{input_csv.stem}.json"
        script_path = detected_output_dir.parent / "script" / f"{dataset_stem}_detect.py"

        context = DetectionContext(
            dataset_stem=dataset_stem,
            input_csv=input_csv,
            ground_truth_csv=ground_truth_csv,
            output_csv=output_csv,
            results_json_path=results_json_path,
            script_path=script_path,
        )

        # Run detection
        context = self.detector.execute(context)
        
        # Read output for analysis
        df_out = pd.read_csv(context.output_csv)
        
        # Analyze error types
        error_type_counts = {}
        for types_str in df_out[df_out['error_detected'] == True]['detected_error_types']:
            if pd.notna(types_str) and types_str:
                for error_type in str(types_str).split('|'):
                    error_type = error_type.strip()
                    if error_type:
                        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        # Compute metrics if ground truth exists
        has_ground_truth = ground_truth_csv is not None
        if has_ground_truth:
            context.evaluation_metrics = compute_metrics_from_output(
                context.output_csv,
                context.ground_truth_csv
            )
        
        # Build results JSON matching README format
        results_json = {
            "model": self.model,
            "temperature": 0.1,  # Could be made configurable
            "dataset": input_csv.stem,
            "task_type": "error_detection",
            "success": True,
            "output_shape": list(df_out.shape),
            "metrics": {
                "has_ground_truth": has_ground_truth,
                "error_type_analysis": {
                    "total_errors_detected": int(df_out['error_detected'].sum()),
                    **error_type_counts
                },
                "total_events": len(df_out),
                "errors_detected": int(df_out['error_detected'].sum()),
            },
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add full metrics if ground truth exists
        if has_ground_truth and context.evaluation_metrics.get("metrics_strict"):
            for style in ["strict", "moderate", "generous"]:
                style_key = f"metrics_{style}"
                if style_key in context.evaluation_metrics:
                    style_data = context.evaluation_metrics[style_key]
                    results_json["metrics"][style.upper()] = {
                        **style_data["metrics"],
                        "true_positives": style_data["confusion_matrix"]["TP"],
                        "false_positives": style_data["confusion_matrix"]["FP"],
                        "true_negatives": style_data["confusion_matrix"]["TN"],
                        "false_negatives": style_data["confusion_matrix"]["FN"],
                        "evaluation_style": style.upper()
                    }
            
            # Add ground truth info
            results_json["metrics"]["ground_truth_errors"] = context.evaluation_metrics.get("metrics_strict", {}).get("counts", {}).get("ground_truth_errors", 0)
        
        # Save results JSON
        results_json_path.write_text(
            json.dumps(results_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return context


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "data"
    detected_output_dir = repo_root / "detected_output"
    results_dir = repo_root / "results"
    scripts_dir = repo_root / "script"
    prompt_dir = repo_root / "prompt"

    # Create necessary directories
    detected_output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    client = OpenAI()

    # Default prompts (used if files don't exist)
    default_detector_prompt = """You are an expert data quality analyst. Your task is to analyze the dataset and detect errors.

Dataset: {dataset_name}
Sample data (first 100 rows):
{dataset_sample}

Please write a COMPLETE, SELF-CONTAINED Python script that:
1. Imports sys, pandas
2. Reads the CSV using: df = pd.read_csv(sys.argv[1])
3. Analyzes each row for data quality errors
4. Creates output dataframe with ALL ORIGINAL COLUMNS plus these TWO additional columns:
   - error_detected: boolean (True if error detected, False otherwise)
   - detected_error_types: string with pipe-separated error types (e.g., "form-based|polluted", or "" if no error)
5. Saves results using: df.to_csv(sys.argv[2], index=False)

IMPORTANT OUTPUT FORMAT: 
- Keep ALL original columns from the input CSV
- Add exactly TWO new columns: error_detected and detected_error_types
- error_detected must be True/False boolean values
- detected_error_types must be pipe-separated string (e.g., "form-based", "polluted|distorted", or "" for no error)

TECHNICAL REQUIREMENTS:
- Use sys.argv[1] for input path and sys.argv[2] for output path
- Do NOT use placeholder variables like INPUT_CSV_PATH or OUTPUT_CSV_PATH
- The script must be executable as: python script.py input.csv output.csv
- Include all necessary imports at the top
- Use pandas 2.0+ compatible code (NO deprecated parameters like infer_datetime_format)
- For pd.to_datetime(), only use: pd.to_datetime(series, errors='coerce', format='mixed')
- Do NOT hardcode file paths - always use sys.argv[1] and sys.argv[2]

Return ONLY the complete Python code wrapped in ```python``` markers. No explanations outside the code block.
"""

    # Try to load from files, fallback to defaults
    detector_prompt = load_prompt_text(
        repo_root, 
        "prompt/vanilla_examples_instructions.txt", 
        default_detector_prompt
    )

    pipeline = DetectionPipeline(
        client=client,
        model=model,
        repo_root=repo_root,
        detector_prompt=detector_prompt,
    )

    csv_files = sorted(data_dir.rglob("*.csv"))
    # Filter out ONLY the separate label files, not data files that happen to have 'label' in their name
    # A label file is one that ONLY contains labels (has _labels, _label, _gt, _groundtruth suffixes)
    csv_files = [f for f in csv_files if not any(
        f.stem.endswith(suffix) for suffix in ['_labels', '_label', '_gt', '_groundtruth', '_withLabel']
    )]
    
    if not csv_files:
        print(f"\nNo CSV files found in {data_dir}")
        print(f"\nSearched for: *.csv files")
        print(f"Excluded: Files ending with _labels.csv, _label.csv, _gt.csv, _groundtruth.csv, _withLabel.csv")
        
        # Show what's actually in the directory
        all_files = list(data_dir.rglob("*"))
        if all_files:
            print(f"\nFiles found in {data_dir}:")
            for f in all_files[:10]:
                print(f"  - {f.name}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more")
        else:
            print(f"\nDirectory {data_dir} is empty or does not exist.")
        return

    print("\n" + "=" * 70)
    print("Error Detection Pipeline")
    print(f"Model: {model} | Datasets: {len(csv_files)}")
    print("=" * 70)

    all_results: List[Dict[str, Any]] = []

    for in_csv in csv_files:
        try:
            print("\n" + "=" * 70)
            print(f"DATASET: {in_csv.name}")
            print("=" * 70)

            context = pipeline.process_dataset(in_csv, detected_output_dir, results_dir)

            result = {
                "dataset": in_csv.name,
                "errors_detected": context.detection_summary.get("errors_detected", 0),
                "total_rows": context.detection_summary.get("total_rows", 0),
                "detection_metrics_strict": context.evaluation_metrics.get("metrics_strict", {}).get("metrics", {}),
                "detection_metrics_moderate": context.evaluation_metrics.get("metrics_moderate", {}).get("metrics", {}),
                "detection_metrics_generous": context.evaluation_metrics.get("metrics_generous", {}).get("metrics", {}),
            }
            all_results.append(result)

            print("\nOUTPUTS:")
            print(f"  - Detected: {context.output_csv}")
            print(f"  - Results JSON: {context.results_json_path}")

            if context.evaluation_metrics:
                print("\nDETECTION METRICS (ALL STYLES):")
                for style in ["strict", "moderate", "generous"]:
                    style_metrics = context.evaluation_metrics.get(f"metrics_{style}", {})
                    m = style_metrics.get("metrics", {})
                    cm = style_metrics.get("confusion_matrix", {})
                    c = style_metrics.get("counts", {})
                    
                    print(f"\n  {style.upper()}:")
                    print(f"    Precision: {m.get('precision', 0):.4f}")
                    print(f"    Recall:    {m.get('recall', 0):.4f}")
                    print(f"    F1 Score:  {m.get('f1_score', 0):.4f}")
                    print(f"    Accuracy:  {m.get('accuracy', 0):.4f}")
                    total_rows = c.get("total_rows", 0) or 1
                    pred_errs = c.get("predicted_errors", 0)
                    print(f"    Errors: {pred_errs} / {total_rows} ({pred_errs / total_rows * 100:.1f}%)")
                    print(
                        f"    TP: {cm.get('TP', 0)} | FP: {cm.get('FP', 0)} | "
                        f"FN: {cm.get('FN', 0)} | TN: {cm.get('TN', 0)}"
                    )

        except Exception as e:
            print(f"\nERROR processing {in_csv.name}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        print("\n" + "=" * 70)
        print(f"SUMMARY - {len(all_results)} DATASETS")
        print("=" * 70)

        total_datasets = len(all_results)
        
        # Calculate averages for each style
        for style in ["strict", "moderate", "generous"]:
            print(f"\n{style.upper()} METRICS:")
            
            avg_det_precision = sum(r[f"detection_metrics_{style}"].get("precision", 0) for r in all_results) / total_datasets
            avg_det_recall = sum(r[f"detection_metrics_{style}"].get("recall", 0) for r in all_results) / total_datasets
            avg_det_f1 = sum(r[f"detection_metrics_{style}"].get("f1_score", 0) for r in all_results) / total_datasets
            
            print(f"  Average Detection:")
            print(f"    Precision: {avg_det_precision:.4f}")
            print(f"    Recall:    {avg_det_recall:.4f}")
            print(f"    F1 Score:  {avg_det_f1:.4f}")

        print("\nINDIVIDUAL RESULTS (GENEROUS):")
        for r in all_results:
            dm = r["detection_metrics_generous"]
            print(f"  {r['dataset']:30s} | Det F1: {dm.get('f1_score', 0):.4f}")

        print("=" * 70)

    print(f"\nComplete - Detected CSV files: {detected_output_dir}")
    print(f"Complete - Results JSON files: {results_dir}")


if __name__ == "__main__":
    main()