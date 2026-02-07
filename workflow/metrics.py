"""
Metrics computation for error detection evaluation
"""

import pandas as pd
from .file_handler import FileHandler


class MetricsCalculator:
    """Calculates evaluation metrics for error detection"""
    
    @staticmethod
    def parse_error_types(types_str):
        """Parse error types string into a set"""
        if pd.isna(types_str) or types_str == '' or types_str is None:
            return set()
        
        types_str = str(types_str).strip().lower()
        
        if '|' in types_str:
            types = types_str.split('|')
        elif ',' in types_str:
            types = types_str.split(',')
        else:
            types = [types_str]
        
        return {t.strip() for t in types if t.strip()}
    
    @staticmethod
    def analyze_detected_error_types(df):
        """Analyze what types of errors the LLM detected"""
        error_type_counts = {
            'form-based': 0,
            'homonymous': 0,
            'synonym': 0,
            'collateral': 0,
            'distorted': 0,
            'polluted': 0
        }
        
        analysis = {
            'total_errors_detected': 0,
            'form_based': 0,
            'homonymous': 0,
            'synonym': 0,
            'collateral': 0,
            'distorted': 0,
            'polluted': 0
        }
        
        error_col = None
        if 'error_detected' in df.columns:
            error_col = 'error_detected'
        elif 'error_flag' in df.columns:
            error_col = 'error_flag'
        
        if error_col is None:
            return analysis
        
        pred_types_col = None
        for col in ['error_types', 'detected_error_types', 'predicted_error_types']:
            if col in df.columns:
                pred_types_col = col
                break
        
        try:
            if df[error_col].dtype == 'object':
                errors_df = df[df[error_col].astype(str).str.lower().isin(['true', '1', 'yes'])]
            else:
                errors_df = df[df[error_col].astype(bool) == True]
            
            analysis['total_errors_detected'] = len(errors_df)
            
        except Exception as e:
            print(f"[DEBUG] Error filtering: {e}")
            return analysis
        
        if pred_types_col and len(errors_df) > 0:
            for idx, row in errors_df.iterrows():
                types = MetricsCalculator.parse_error_types(row[pred_types_col])
                types_lower = {t.lower().replace('_', '-').replace(' ', '-') for t in types}
                
                if 'form-based' in types_lower or 'formbased' in types_lower:
                    error_type_counts['form-based'] += 1
                if 'homonymous' in types_lower or 'homonym' in types_lower:
                    error_type_counts['homonymous'] += 1
                if 'synonym' in types_lower or 'synonymous' in types_lower:
                    error_type_counts['synonym'] += 1
                if 'collateral' in types_lower:
                    error_type_counts['collateral'] += 1
                if 'distorted' in types_lower or 'distortion' in types_lower:
                    error_type_counts['distorted'] += 1
                if 'polluted' in types_lower or 'pollution' in types_lower:
                    error_type_counts['polluted'] += 1
            
            analysis.update({
                'form_based': error_type_counts['form-based'],
                'homonymous': error_type_counts['homonymous'],
                'synonym': error_type_counts['synonym'],
                'collateral': error_type_counts['collateral'],
                'distorted': error_type_counts['distorted'],
                'polluted': error_type_counts['polluted']
            })
        
        return analysis
    
    @staticmethod
    def compute_metrics(output_path, original_csv_path=None):
        """Compute evaluation metrics if ground truth exists, otherwise just detection info
        
        Checks three sources for ground truth labels:
        1. Labels in the output file itself
        2. Separate labels file (e.g., dataset_labels.csv)
        3. Labels in the original input file
        
        Note: Ground truth can be either:
        - Boolean label column (label, is_error, etc.) indicating if row has error
        - Error type column (error_type, error_types) directly specifying error types
        """
        metrics = {}
        
        try:
            df = pd.read_csv(output_path)
            
            error_col = 'error_detected' if 'error_detected' in df.columns else 'error_flag'
            
            if error_col not in df.columns:
                return metrics
            
            # Add error type analysis
            error_type_analysis = MetricsCalculator.analyze_detected_error_types(df)
            metrics['error_type_analysis'] = error_type_analysis
            
            # Check for ground truth - first look for error type column (primary method)
            gt_types_col = None
            for col in ['error_type', 'error_types', 'gt_error_type', 'label']:
                if col in df.columns:
                    gt_types_col = col
                    break
            
            # Also check for boolean label column
            label_col = None
            for col in ['is_error', 'error', 'ground_truth', 'gt', 'has_error']:
                if col in df.columns:
                    label_col = col
                    break
            
            # If no ground truth in output file, check for separate labels file
            if gt_types_col is None and label_col is None and original_csv_path:
                labels_file = FileHandler.find_labels_file(original_csv_path)
                if labels_file:
                    df, merge_success = FileHandler.merge_labels_with_data(df, labels_file)
                    if merge_success:
                        # Re-check for columns after merge
                        for col in ['error_type', 'error_types', 'gt_error_type', 'label']:
                            if col in df.columns:
                                gt_types_col = col
                                print(f"[INFO] Using error types from separate file: {labels_file}")
                                break
                        
                        if gt_types_col is None:
                            for col in ['is_error', 'error', 'ground_truth', 'gt', 'has_error']:
                                if col in df.columns:
                                    label_col = col
                                    print(f"[INFO] Using labels from separate file: {labels_file}")
                                    break
            
            # If we have error types column, derive boolean labels from it
            if gt_types_col is not None:
                # Create boolean ground truth from error types
                # If error_type is empty/NaN, it's not an error
                ground_truth = df[gt_types_col].notna() & (df[gt_types_col] != '') & (df[gt_types_col].astype(str).str.strip() != '')
                metrics['has_ground_truth'] = True
            elif label_col is not None:
                ground_truth = df[label_col].astype(bool)
                metrics['has_ground_truth'] = True
            else:
                # No ground truth - just return detection statistics
                metrics['has_ground_truth'] = False
                metrics['total_events'] = len(df)
                metrics['errors_detected'] = int(df[error_col].sum())
                return metrics
            
            # Find predicted error types column
            pred_types_col = None
            for col in ['detected_error_types', 'predicted_error_types']:
                if col in df.columns and col != gt_types_col:
                    pred_types_col = col
                    break
            
            predictions = df[error_col].astype(bool)
            
            if gt_types_col and pred_types_col:
                for style in ['STRICT', 'MODERATE', 'GENEROUS']:
                    style_metrics = MetricsCalculator._compute_metrics_by_style(
                        df, ground_truth, predictions, gt_types_col, pred_types_col, style
                    )
                    metrics[style] = style_metrics
            else:
                basic_metrics = MetricsCalculator._compute_basic_metrics(ground_truth, predictions, len(df))
                metrics['BASIC'] = basic_metrics
            
            metrics.update({
                'ground_truth_column': gt_types_col if gt_types_col else label_col,
                'gt_types_column': gt_types_col,
                'pred_types_column': pred_types_col,
                'total_events': len(df),
                'errors_detected': int(predictions.sum()),
                'ground_truth_errors': int(ground_truth.sum())
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to compute metrics: {e}")
        
        return metrics
    
    @staticmethod
    def _compute_basic_metrics(ground_truth, predictions, total):
        """Compute basic binary classification metrics"""
        tp = int(((ground_truth == True) & (predictions == True)).sum())
        fp = int(((ground_truth == False) & (predictions == True)).sum())
        tn = int(((ground_truth == False) & (predictions == False)).sum())
        fn = int(((ground_truth == True) & (predictions == False)).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'accuracy': round(accuracy, 4),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    @staticmethod
    def _compute_metrics_by_style(df, ground_truth, predictions, gt_types_col, pred_types_col, style):
        """Compute metrics based on evaluation style
        
        STRICT: Predicted types must exactly match ground truth types
        MODERATE: Predicted types must be a subset of ground truth types 
                  (all predicted must be correct, but can miss some GT types)
        GENEROUS: At least one predicted type must overlap with ground truth types
        """
        tp = fp = tn = fn = 0
        
        for idx, row in df.iterrows():
            gt_error = ground_truth.iloc[idx]
            pred_error = predictions.iloc[idx]
            
            if not pred_error and not gt_error:
                tn += 1
            elif not pred_error and gt_error:
                fn += 1
            elif pred_error and not gt_error:
                fp += 1
            else:
                gt_types = MetricsCalculator.parse_error_types(row[gt_types_col])
                pred_types = MetricsCalculator.parse_error_types(row[pred_types_col])
                
                if style == 'STRICT':
                    # Exact match required
                    if gt_types == pred_types:
                        tp += 1
                    else:
                        fp += 1
                elif style == 'MODERATE':
                    # Predicted types must be a subset of ground truth types
                    # All predicted types must be correct, but can miss some GT types
                    if pred_types.issubset(gt_types) and len(pred_types) > 0:
                        tp += 1
                    else:
                        fp += 1
                elif style == 'GENEROUS':
                    # At least one overlap required
                    if len(gt_types & pred_types) > 0:
                        tp += 1
                    else:
                        fp += 1
        
        total = len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'accuracy': round(accuracy, 4),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'evaluation_style': style
        }