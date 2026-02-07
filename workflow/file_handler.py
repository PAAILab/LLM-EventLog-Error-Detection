"""
File handling operations for error detection workflow
"""

import pandas as pd
from pathlib import Path


class FileHandler:
    """Handles file operations for datasets and labels"""
    
    @staticmethod
    def find_datasets(data_dir):
        """Find all CSV files in the data directory"""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"[WARNING] Data directory '{data_dir}' not found")
            return []
        
        csv_files = list(data_dir.glob("**/*.csv"))
        print(f"[INFO] Found {len(csv_files)} CSV files in {data_dir}")
        return csv_files
    
    @staticmethod
    def extract_dataset_name(csv_path):
        """Extract dataset name from CSV path"""
        csv_path = Path(csv_path)
        dataset_name = csv_path.stem
        return dataset_name
    
    @staticmethod
    def create_output_path(csv_path, output_dir):
        """Create output path for detected dataset"""
        dataset_name = FileHandler.extract_dataset_name(csv_path)
        output_filename = f"{dataset_name}_detected.csv"
        output_path = Path(output_dir) / output_filename
        return output_path, dataset_name
    
    @staticmethod
    def find_labels_file(csv_path):
        """Find corresponding labels file if it exists
        
        Looks for files named:
        - {dataset_name}_labels.csv
        - {dataset_name}_gt.csv
        - {dataset_name}_groundtruth.csv
        
        Returns path to labels file or None if not found
        """
        csv_path = Path(csv_path)
        dataset_name = csv_path.stem
        data_dir = csv_path.parent
        
        # Try different naming conventions
        label_patterns = [
            f"{dataset_name}_labels.csv",
            f"{dataset_name}_gt.csv",
            f"{dataset_name}_groundtruth.csv",
            f"{dataset_name}_label.csv"
        ]
        
        for pattern in label_patterns:
            label_path = data_dir / pattern
            if label_path.exists():
                print(f"[INFO] Found separate labels file: {label_path}")
                return label_path
        
        return None
    
    @staticmethod
    def merge_labels_with_data(data_df, labels_path):
        """Merge labels from separate file with data
        
        Assumes both files have matching row indices or a common key column
        Supports both error_type column (primary) and boolean label columns
        """
        try:
            labels_df = pd.read_csv(labels_path)
            print(f"[INFO] Labels file shape: {labels_df.shape}")
            print(f"[INFO] Labels file columns: {list(labels_df.columns)}")
            
            # Check if labels file has matching number of rows
            if len(labels_df) != len(data_df):
                print(f"[WARNING] Row count mismatch: data={len(data_df)}, labels={len(labels_df)}")
            
            # First, look for error_type column (primary method)
            types_col = None
            for col in ['error_type', 'error_types', 'gt_error_type', 'label']:
                if col in labels_df.columns:
                    types_col = col
                    break
            
            # Also check for boolean label column
            label_col = None
            for col in ['is_error', 'error', 'ground_truth', 'gt', 'has_error']:
                if col in labels_df.columns:
                    label_col = col
                    break
            
            if types_col is None and label_col is None:
                print(f"[WARNING] No label or error_type column found in labels file")
                return data_df, False
            
            # Merge based on index (assume same order)
            merged_df = data_df.copy()
            
            columns_merged = []
            if types_col:
                merged_df[types_col] = labels_df[types_col]
                columns_merged.append(types_col)
            
            if label_col:
                merged_df[label_col] = labels_df[label_col]
                columns_merged.append(label_col)
            
            print(f"[INFO] Merged columns: {', '.join(columns_merged)}")
            
            return merged_df, True
            
        except Exception as e:
            print(f"[ERROR] Failed to merge labels: {e}")
            return data_df, False
    
    @staticmethod
    def load_prompt(prompt_path):
        """Load prompt from specified file"""
        prompt_path = Path(prompt_path)
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file '{prompt_path}' not found")
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    
    @staticmethod
    def check_output_files(output_path):
        """Check if the expected output files were created and have required columns"""
        print(f"[DEBUG] Checking for output file: {output_path}")
        
        if output_path.exists():
            try:
                df = pd.read_csv(output_path)
                print(f"[DEBUG] Output file found! Shape: {df.shape}")
                print(f"[DEBUG] Columns in output: {list(df.columns)}")
                
                # Check for error detection column
                error_col = None
                if 'error_detected' in df.columns:
                    error_col = 'error_detected'
                elif 'error_flag' in df.columns:
                    error_col = 'error_flag'
                
                if error_col is None:
                    print(f"[ERROR] MISSING REQUIRED COLUMN: Neither 'error_detected' nor 'error_flag' found")
                    return False, output_path, None
                else:
                    errors_detected = df[error_col].sum()
                    print(f"[DEBUG] Error detection column '{error_col}' found: {errors_detected} errors")
                
                # Check for error types column
                types_col = None
                if 'detected_error_types' in df.columns:
                    types_col = 'detected_error_types'
                elif 'error_types' in df.columns:
                    types_col = 'error_types'
                
                if types_col is None:
                    print(f"[WARNING] Neither 'detected_error_types' nor 'error_types' column found")
                else:
                    print(f"[DEBUG] Error types column '{types_col}' found")
                
                return True, output_path, df.shape
            except Exception as e:
                print(f"[DEBUG] Error reading output file: {e}")
                return False, output_path, None
        else:
            print(f"[DEBUG] Output file NOT found at: {output_path}")
        
        return False, output_path, None