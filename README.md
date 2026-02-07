# LLM Event Log Error Detection Framework

A Python tool for automated error detection in event log datasets using GPT models. Processes CSV datasets, generates detection code using LLMs, and computes evaluation metrics.

## Features

- LLM-powered error detection code generation
- Automated metrics computation (precision, recall, F1-score, accuracy)
- Three evaluation styles: STRICT, MODERATE, GENEROUS
- Batch processing of multiple datasets
- JSON result logging with full metrics

## Directory Structure

```
.
├── data/                    # Input datasets (CSV files)
├── prompt/                  # Prompt templates
├── script/                  # Generated detection scripts
├── detected_output/         # Detection results (CSV files)
├── results/                 # JSON summaries with metrics
├── workflow/                # Breakdown of code with separate functionality
└── main.py                  # Main script
```

## Installation

```bash
pip install pandas openai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

```bash
python main.py
```

### With Options

```bash
python main.py \
    --prompt-file prompt/custom_prompt.txt \
    --temperature 0.5 \
    --model gpt-5.2 \
    --data-dir data/
```

### Command-Line Arguments

| Argument        | Default                       | Description                          |
| --------------- | ----------------------------- | ------------------------------------ |
| `--prompt-file` | `prompt/vanilla_examples.txt` | Path to prompt file                  |
| `--temperature` | `0.3`                         | Temperature for generation (0.0-1.0) |
| `--model`       | `gpt-5.2`                     | GPT model to use                     |
| `--data-dir`    | `data`                        | Directory containing datasets        |
| `--api-key`     | (env var)                     | OpenAI API key                       |

## Input Format

The tool processes CSV files from the `data/` directory. Ground truth labels can be provided in three different ways:

### Dataset Format Options

#### Option 1: Dataset WITH Ground Truth Labels in Same File (Recommended)

Your CSV contains both the event log data AND ground truth error types in the same file:

```csv
case_id,activity,timestamp,resource,error_type
1,Submit Application,2024-01-01 10:00:00,Clerk-001,
1,Review Application,2024-01-01 11:00:00,Manager-003,
2,Submitt Application,2024-01-01 12:00:00,Clerk-002,form-based
2,Process Payment_X7Q2,2024-01-01 13:00:00,System-001,polluted
3,Approve Request,2024-01-01 14:00:00,Manager-002,form-based|polluted
```

**Required columns for evaluation**:

- Event log data: `case_id`, `activity`, `timestamp`, `resource` (or similar)
- **Ground truth error types**: Column named `error_type`, `error_types`, `gt_error_type`, or `label`
  - Empty or blank = no error (clean event)
  - Contains error type(s) = error present (e.g., "form-based", "polluted", "form-based|polluted")
  - Multiple types should be pipe-separated: `form-based|polluted`

**Output**: Full metrics (STRICT, MODERATE, GENEROUS) with precision, recall, F1, accuracy

#### Option 2: Dataset WITHOUT Ground Truth Labels (Detection Only)

Your CSV contains only the event log data without any error type labels:

```csv
case_id,activity,timestamp,resource
1,Submit Application,2024-01-01 10:00:00,Clerk-001
1,Review Application,2024-01-01 11:00:00,Manager-003
2,Submitt Application,2024-01-01 12:00:00,Clerk-002
2,Process Payment_X7Q2,2024-01-01 13:00:00,System-001
```

**Required columns**:

- Event log data: `case_id`, `activity`, `timestamp`, `resource` (or similar)
- **No label columns needed**

**Output**: Detection statistics only (total errors detected, error type distribution)

#### Option 3: Dataset with Separate Labels File

Your event data and ground truth error types are in **two separate files** in the same directory:

**Event data file** (`my_dataset.csv`):

```csv
case_id,activity,timestamp,resource
1,Submit Application,2024-01-01 10:00:00,Clerk-001
1,Review Application,2024-01-01 11:00:00,Manager-003
2,Submitt Application,2024-01-01 12:00:00,Clerk-002
3,Process Payment_X7Q2,2024-01-01 13:00:00,System-001
```

**Labels file** (`my_dataset_labels.csv`):

```csv
error_type

form-based
polluted
```

**Naming conventions** - The labels file must be named using one of these patterns:

- `{dataset_name}_labels.csv`
- `{dataset_name}_label.csv`
- `{dataset_name}_gt.csv`
- `{dataset_name}_groundtruth.csv`

**Example**: If your data file is `credit_data.csv`, the labels file should be `credit_data_labels.csv`

**Important**:

- Both files must have the **same number of rows**
- Rows are matched by **index** (first row in data matches first row in labels)
- Labels file must contain `error_type` (or `error_types`, `gt_error_type`, `label`)
- Empty/blank cells in error_type column = no error (clean event)
- Non-empty cells = error present with specified type(s)

**Output**: Full metrics (STRICT, MODERATE, GENEROUS) with precision, recall, F1, accuracy

### Important Notes

- **Automatic detection**: The tool automatically detects which option applies to your dataset
- **Error type is the label**: Unlike binary classification, the error_type column directly contains the error category (not True/False)
- **Empty means clean**: Blank or empty error_type cells indicate clean events (no errors)
- **Column name flexibility**: Error type columns are auto-detected if named `error_type`, `error_types`, `gt_error_type`, or `label`
- **Multiple error types**: Use pipe-separated format (e.g., `form-based|polluted`)

### Complete Examples

**Option 1 - Error Types in Same File**:

```csv
case_id,activity,timestamp,resource,error_type
101,Submit Application,2024-01-15 09:00:00,Clerk-001,
101,Review Application,2024-01-15 10:30:00,Manager-003,
102,Submitt Application,2024-01-15 11:00:00,Clerk-002,form-based
103,Approve Request_X8Q2,2024-01-15 12:00:00,Manager-002,polluted
104,Process Payment,2024-01-15 13:00:00,System-001,form-based|polluted
105,Close Case,2024-01-15 14:00:00,Clerk-003,
```

**Option 2 - No Labels** (for unlabeled data):

```csv
case_id,activity,timestamp,resource
101,Submit Application,2024-01-15 09:00:00,Clerk-001
102,Submitt Application,2024-01-15 11:00:00,Clerk-002
103,Approve Request_X8Q2,2024-01-15 12:00:00,Manager-002
```

**Option 3 - Separate Files**:

File: `data/my_events.csv`

```csv
case_id,activity,timestamp,resource
101,Submit Application,2024-01-15 09:00:00,Clerk-001
102,Submitt Application,2024-01-15 11:00:00,Clerk-002
103,Approve Request_X8Q2,2024-01-15 12:00:00,Manager-002
104,Close Case,2024-01-15 13:00:00,Clerk-003
```

File: `data/my_events_labels.csv`

```csv
error_type

form-based
polluted

```

**Note**: Row 1 and Row 4 are blank (clean events), Row 2 and Row 3 have error types

## Prompt Engineering

The prompt file is the core component that instructs the LLM how to generate error detection code. A well-constructed prompt should include multiple types of information to guide the model effectively.

### Prompt Structure

An effective prompt should contain the following sections:

#### 1. Dataset Format Information

Describe the structure and format of your input data:

```
DATASET FORMAT:
- Format: CSV file with event log data
- Columns: case_id, activity, timestamp, resource, cost
- Total events: Varies by dataset
- Expected output: CSV with original columns + error_detected + detected_error_types
```

**Purpose**: Helps the LLM understand the data structure and generate appropriate column references.

#### 2. Process Mining Specific Information

Include domain knowledge about process mining and event logs:

```
PROCESS MINING CONTEXT:
- Each row represents an event in a business process
- Events are grouped by case_id (process instance)
- Events should follow a logical sequence
- Activities should be consistent within their naming
- Timestamps should be chronological within each case
- Resources should be valid and consistent
```

**Purpose**: Provides context about process mining principles that inform error detection logic.

#### 3. Error Type Definitions

Define each error type clearly with specific characteristics:

```
ERROR TYPES:
1. FORM-BASED: Syntactic errors in activity names
   - Typos, misspellings, extra spaces
   - Inconsistent capitalization
   - Example: "Submitt Application" instead of "Submit Application"

2. HOMONYMOUS: Same name for different activities
   - Identical labels for semantically different tasks
   - Example: "Review" used for both document review and supervisor review

3. SYNONYM: Different names for the same activity
   - Multiple labels for identical tasks
   - Example: "Approve", "Accept", "Grant" all meaning approval

4. COLLATERAL: Events that should not exist together
   - Logically incompatible activities in same case
   - Example: "Reject Application" followed by "Process Payment"

5. DISTORTED: Attribute value errors
   - Invalid or out-of-range values
   - Example: Negative cost, future timestamps, impossible durations

6. POLLUTED: Infrequent or anomalous patterns
   - Rare activity sequences
   - Statistical outliers in timing or frequency
```

**Purpose**: Provides clear definitions that the LLM can translate into detection rules.

#### 4. Dataset-Specific Information

Include characteristics unique to the current dataset:

```
DATASET SPECIFICS:
- Name: loan_applications_2024.csv
- Cases: 500 process instances
- Events per case: Average 8-12 events
- Time range: Jan 2024 - Dec 2024
- Known issues: Some activities have trailing spaces
- Expected error rate: 5-10% of events
```

**Purpose**: Helps calibrate detection sensitivity and handle dataset-specific quirks.

#### 5. Examples

Provide concrete examples of errors and correct patterns:

```
EXAMPLES:

Example 1: Form-based error
Event: case_123, "Submitt Application", 2024-01-15, clerk_01
Issue: Typo in activity name
Correct: "Submit Application"
Detection: error_detected=True, detected_error_types="form-based"

Example 2: Collateral error
Event: case_456, "Disburse Loan", 2024-02-20, manager_01
Context: Previous event was "Reject Application"
Issue: Disbursement after rejection is illogical
Detection: error_detected=True, detected_error_types="collateral"

Example 3: No error
Event: case_789, "Submit Application", 2024-03-10, clerk_03
Issue: None - valid activity, resource, and sequence
Detection: error_detected=False, detected_error_types=""
```

**Purpose**: Gives the LLM concrete patterns to recognize and replicate.

#### 6. Output Requirements

Specify exact output format requirements:

```
OUTPUT REQUIREMENTS:
1. Generate Python code that reads the CSV file from command line argument
2. Create two new columns:
   - error_detected: Boolean (True/False)
   - detected_error_types: String with pipe-separated types (e.g., "form-based|polluted")
3. Preserve all original columns
4. Save output to: detected_output/{dataset_name}_detected.csv
5. Print summary statistics
```

**Purpose**: Ensures the generated code produces output in the expected format.

### Example Prompt Template

Here's a complete prompt template combining all sections:

```
TASK: Generate Python code to detect errors in event log data

DATASET FORMAT:
- CSV with columns: case_id, activity, timestamp, resource
- Each row is a process event
- Output: Same CSV + error_detected + detected_error_types columns

ERROR TYPES:
[Include definitions as shown above]

DATASET SPECIFICS:
[Current dataset characteristics]

EXAMPLES: (Optional)
[Concrete error and non-error examples]

OUTPUT REQUIREMENTS:
[Exact specifications for output format]

Generate Python code that implements this error detection logic.
```

### Tips for Effective Prompts

1. **Be Specific**: Vague definitions lead to inconsistent detection
2. **Include Edge Cases**: Show examples of borderline cases
3. **Update for Each Dataset**: Modify dataset-specific section for each new dataset
4. **Balance Strictness**: Too strict = many false negatives, too loose = many false positives
5. **Test Iteratively**: Run on sample data and refine definitions based on results

### Using the Prompt File

Save your prompt in the `prompt/` directory and reference it:

```bash
python main.py --prompt-file prompt/my_custom_prompt.txt
```

### Prompt Variants: Vanilla Examples vs Vanilla Examples Instructions

The tool supports different prompt engineering approaches, with two main variants designed to test different levels of guidance:

#### Vanilla

**Description**: A base prompt that provides only error type definitions.

**Content**:

- Brief task description
- Error type definitions (what each error type is)
- Output format requirements

**Motive**: Test the LLM's ability to infer detection logic from definitions only, without giving any supervised data.

#### Vanilla Examples

**Description**: A minimal prompt that provides only error type definitions and concrete examples without detailed detection instructions.

**Content**:

- Brief task description
- Error type definitions (what each error type is)
- Concrete examples of each error type
- Output format requirements

**Motive**: Tests the LLM's ability to infer detection logic from definitions and examples alone, without explicit step-by-step instructions. This approach assumes the model can learn patterns from examples and generalize them.

**Best for**:

- Quick prototyping
- Datasets similar to the examples provided
- When you want to test the model's baseline capabilities

**Example structure**:

```
TASK: Detect errors in event logs

ERROR TYPES:
1. Form-based: Multiple events with same timestamp
   Example: [show examples]

2. Polluted: Activities with machine-generated suffixes
   Example: [show examples]

OUTPUT: CSV with error_detected and detected_error_types columns
```

#### Vanilla Examples Instructions

**Description**: An enhanced prompt that includes error definitions, examples, AND explicit detection instructions for each error type.

**Content**:

- Task description
- Error type definitions
- **Detailed detection instructions** (how to detect each type)
- Concrete examples
- Detection criteria and thresholds
- Output format requirements

**Motive**: Guides the LLM with explicit step-by-step detection logic, reducing ambiguity and improving consistency. This approach provides algorithmic guidance while still allowing the model to implement the logic.

**Best for**:

- Production environments requiring reliability
- Complex error types that need specific criteria
- When you want more control over detection behavior

**Example structure**:

```
TASK: Detect errors in event logs

ERROR TYPES:
1. Form-based: Multiple events with same timestamp

   DETECTION INSTRUCTIONS:
   - Identify timestamp clusters within each case
   - Cluster must have ≥2 events with identical timestamps
   - Events must have different activity labels
   - Mark all events in cluster as form-based errors

   Example: [show examples]

2. Polluted: Activities with machine-generated suffixes

   DETECTION INSTRUCTIONS:
   - Check if activity ends with underscore-separated suffix
   - Suffix pattern: _[alphanumeric]_[timestamp]
   - Alphanumeric token is 5-12 characters
   - Extract base label by removing suffix

   Example: [show examples]
```

#### When to Add Domain/Dataset-Specific Information

**Domain Knowledge**:
When you have business rules specific to your industry (e.g., healthcare, finance), adding domain-specific sections dramatically improves accuracy:

```
DOMAIN: Healthcare Patient Admissions
BUSINESS RULES:
- Patient registration must occur before diagnosis
- Discharge cannot occur before treatment completion
- Valid diagnoses: [list of medical codes]
- Typical visit duration: 30 minutes to 4 hours
```

**Dataset-Specific Information**:
When you know characteristics of your specific dataset, include them for calibration:

```
DATASET SPECIFICS:
- Known issue: Activities have trailing spaces
- Expected error rate: ~8% of events
- Most common errors: form-based (60%), polluted (25%)
- Time period: Q1 2024
```

**Dataset Snapshots**:
Including actual data samples with labeled errors helps the LLM understand the real distribution:

```
REAL EXAMPLES FROM DATASET:

Clean events:
101,Submit Application,2024-01-15 09:00:00,Clerk-001,
101,Review Application,2024-01-15 10:30:00,Manager-003,

Errors found:
102,Submitt Application_X7aQ2_20240115 091500,2024-01-15 09:15:00,Clerk-002,
    ^ form-based error (typo) + polluted (suffix)

103,Process Payment,2024-01-15 14:00:00,System-001,
103,Process Payment,2024-01-15 14:00:01,System-001,
    ^ collateral errors (duplicate within 1 second)
```

**Impact**: Domain knowledge + dataset specifics can improve F1-score by 15-25% compared to vanilla prompts alone.

**Getting Started**: Use the provided prompts in the `prompt/` directory as a base template. You can customize them by adding the domain-specific, dataset-specific, and example sections mentioned above to improve detection accuracy for your particular use case.

## Output Format

### Detected Datasets

Files saved in `detected_output/` as `{dataset_name}_detected.csv` containing:

- All original columns
- `error_detected`: Boolean (True/False)
- `detected_error_types`: Pipe-separated types

### Results JSON

Saved in `results/` as `{model}_detection_{dataset}.json`

#### With Ground Truth Labels

When your dataset includes a ground truth column (`label`, `is_error`, etc.), the JSON includes full evaluation metrics:

```json
{
  "model": "gpt-5.2",
  "temperature": 0.3,
  "dataset": "my_dataset",
  "task_type": "error_detection",
  "success": true,
  "output_shape": [1000, 15],
  "metrics": {
    "has_ground_truth": true,
    "error_type_analysis": {
      "total_errors_detected": 92,
      "form_based": 45,
      "homonymous": 12,
      "synonym": 8,
      "collateral": 15,
      "distorted": 7,
      "polluted": 5
    },
    "STRICT": {
      "precision": 0.85,
      "recall": 0.72,
      "f1_score": 0.78,
      "accuracy": 0.91,
      "true_positives": 72,
      "false_positives": 13,
      "true_negatives": 887,
      "false_negatives": 28,
      "evaluation_style": "STRICT"
    },
    "MODERATE": {
      "precision": 0.88,
      "recall": 0.75,
      "f1_score": 0.81,
      "accuracy": 0.92,
      "true_positives": 75,
      "false_positives": 10,
      "true_negatives": 890,
      "false_negatives": 25,
      "evaluation_style": "MODERATE"
    },
    "GENEROUS": {
      "precision": 0.92,
      "recall": 0.85,
      "f1_score": 0.88,
      "accuracy": 0.94,
      "true_positives": 85,
      "false_positives": 7,
      "true_negatives": 893,
      "false_negatives": 15,
      "evaluation_style": "GENEROUS"
    },
    "ground_truth_column": "label",
    "gt_types_column": "error_type",
    "pred_types_column": "detected_error_types",
    "total_events": 1000,
    "errors_detected": 92,
    "ground_truth_errors": 100
  },
  "error": null,
  "timestamp": "2024-02-07T10:30:00.123456"
}
```

#### Without Ground Truth Labels

When no ground truth column is present, the JSON includes only detection statistics:

```json
{
  "model": "gpt-5.2",
  "temperature": 0.3,
  "dataset": "unlabeled_dataset",
  "task_type": "error_detection",
  "success": true,
  "output_shape": [1000, 12],
  "metrics": {
    "has_ground_truth": false,
    "error_type_analysis": {
      "total_errors_detected": 87,
      "form_based": 40,
      "homonymous": 15,
      "synonym": 10,
      "collateral": 12,
      "distorted": 6,
      "polluted": 4
    },
    "total_events": 1000,
    "errors_detected": 87
  },
  "error": null,
  "timestamp": "2024-02-07T10:30:00.123456"
}
```

**Key Differences:**

- **With labels**: Includes STRICT, MODERATE, GENEROUS metrics with precision/recall/F1/accuracy
- **Without labels**: Only includes error counts and type distribution
- Both include `error_type_analysis` showing breakdown of detected error types

## Evaluation Styles

### STRICT

Predicted error types must **exactly match** ground truth types.

**Example:**

- GT: `form-based|polluted` → Pred: `form-based|polluted` ✓
- GT: `form-based|polluted` → Pred: `form-based` ✗

### MODERATE

Predicted error types must be a **subset** of ground truth types (all predicted must be correct, can miss some).

**Example:**

- GT: `form-based|polluted` → Pred: `form-based` ✓
- GT: `form-based` → Pred: `form-based|polluted` ✗

### GENEROUS

At least **one predicted type** must match ground truth.

**Example:**

- GT: `form-based|polluted` → Pred: `form-based` ✓
- GT: `form-based|polluted` → Pred: `synonym` ✗

## Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / Total

Where:

- **TP**: True Positives (correctly detected errors)
- **FP**: False Positives (incorrectly flagged as errors)
- **TN**: True Negatives (correctly identified as non-errors)
- **FN**: False Negatives (missed errors)

## Error Types

The system recognizes these error types:

- `form-based`
- `homonymous`
- `synonym`
- `collateral`
- `distorted`
- `polluted`

## Examples

```bash
# Process all datasets with default settings
python main.py

# Use custom prompt
python main.py --prompt-file prompt/my_prompt.txt

# Use different model
python main.py --model gpt-4-turbo

# Custom temperature
python main.py --temperature 0.5
```

## Troubleshooting

**API Key Error:**

```
Error: OpenAI API token not found
```

Solution: Set `OPENAI_API_KEY` environment variable

**No Datasets Found:**

```
Warning: Data directory 'data' not found
```

Solution: Create `data/` directory and add CSV files

**Model Not Found:**

```
Model 'gpt-x' does not exist
```

Solution: Use valid model names (gpt-5.2, gpt-4-turbo, etc.)
