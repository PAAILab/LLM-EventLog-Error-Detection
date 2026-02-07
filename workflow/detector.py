"""
Main GPT Detector Processor
Orchestrates the error detection workflow
"""

import os
import json
from pathlib import Path
from datetime import datetime
from .code_generator import CodeGenerator
from .metrics import MetricsCalculator
from .file_handler import FileHandler
from .utils import sanitize_python_code, validate_python_syntax, extract_python_code, log_clean, log_error


class GPTDetectorProcessor:
    """Main processor for GPT-based error detection"""
    
    def __init__(self, api_key=None, model_name="gpt-5.2"):
        """Initialize the GPT Detector Processor"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API token not found. Please set OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        
        # Set up directories
        self.data_dir = Path('data')
        self.script_dir = Path('script')
        self.results_dir = Path('results')
        self.output_dir = Path('detected_output')
        
        # Create directories if they don't exist
        self.script_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize code generator
        self.code_generator = CodeGenerator(self.api_key, self.model_name, self.script_dir)
    
    def find_datasets(self):
        """Find all CSV files in the data directory"""
        return FileHandler.find_datasets(self.data_dir)
    
    def save_result_json(self, dataset_name, prompt_content, response, 
                        output_shape, success, temperature, metrics, error_msg=None):
        """Save detection results in JSON format"""
        model_short = self.model_name.split('/')[-1].replace('.', '_').replace('-', '_')
        result_file = self.results_dir / f"{model_short}_detection_{dataset_name}.json"
        
        result_data = {
            "model": self.model_name,
            "temperature": temperature,
            "dataset": dataset_name,
            "task_type": "error_detection",
            "prompt": prompt_content[:300] + "..." if len(prompt_content) > 300 else prompt_content,
            "response": response[:300] + "..." if len(response) > 300 else response,
            "success": success,
            "output_shape": list(output_shape) if output_shape else None,
            "metrics": metrics if metrics else {},
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"[INFO] Results saved to: {result_file}")
        return result_file
    
    def process_dataset(self, csv_file, prompt_file, temperature=0.3):
        """Process a single dataset with error detection"""
        csv_path = Path(csv_file)
        prompt_path = Path(prompt_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        output_path, dataset_name = FileHandler.create_output_path(csv_path, self.output_dir)
        
        log_clean(f"Processing dataset={dataset_name}, model={self.model_name}")
        
        error_msg = None
        metrics = {}
        
        try:
            # Load prompt as-is without modification
            prompt_content = FileHandler.load_prompt(prompt_path)
            
            # Generate code using the prompt as-is
            response = self.code_generator.generate_code(prompt_content, temperature=temperature)
            if response is None:
                error_msg = "Failed to generate code"
                raise Exception(error_msg)
            
            log_clean("Code generated successfully")
            
            # Extract and sanitize
            code = extract_python_code(response)
            if not code:
                error_msg = "Failed to extract code from response"
                raise Exception(error_msg)
            
            code = sanitize_python_code(code)
            
            # Validate syntax
            is_valid, syntax_error = validate_python_syntax(code)
            if not is_valid:
                error_msg = f"Generated code has syntax errors: {syntax_error}"
                print(f"[ERROR] {error_msg}")
                print(f"[DEBUG] Code preview:\n{code[:500]}")
                raise ValueError(error_msg)
            
            if "```" in code:
                error_msg = "Sanitization failed: markdown fences still present"
                raise ValueError(error_msg)
            
            # Save script
            script_path = self.code_generator.save_generated_script(code, dataset_name)
            if script_path is None:
                error_msg = "Failed to save generated script"
                raise Exception(error_msg)
            
            log_clean(f"Script saved to: {script_path}")
            
            # Run script
            success, stdout, stderr, result = self.code_generator.run_script(script_path, csv_path)
            
            if stderr or (stdout and 'error' in stdout.lower()):
                error_indicators = ['traceback', 'exception', 'error', 'syntaxerror']
                combined_output = (stdout + stderr).lower()
                
                if any(indicator in combined_output for indicator in error_indicators):
                    log_error(f"Script execution failed. Check logs.")
                    error_msg = f"Script execution failed"
                    success = False
            
            # Check output
            output_created, final_output_path, output_shape = FileHandler.check_output_files(output_path)
            
            if output_created:
                log_clean("Computing metrics and analyzing error types...")
                metrics = MetricsCalculator.compute_metrics(output_path, csv_path)
            else:
                if not error_msg:
                    error_msg = f"Output file not created"
                metrics = {}
            
            # Save results
            result_file = self.save_result_json(
                dataset_name, prompt_content, response, 
                output_shape, success and output_created, temperature, metrics, error_msg
            )
            
            if success and output_created:
                log_clean(f"Detection completed! Output: {final_output_path}")
                if metrics and 'error_type_analysis' in metrics:
                    analysis = metrics['error_type_analysis']
                    print(f"[INFO] Detected {analysis['total_errors_detected']} errors")
                if metrics.get('has_ground_truth'):
                    print(f"[INFO] Ground truth available - computed evaluation metrics")
                else:
                    print(f"[INFO] No ground truth - detection statistics only")
            else:
                log_error(f"Detection failed: {error_msg}")
            
            return success and output_created, script_path, final_output_path if output_created else None, metrics
            
        except Exception as e:
            if not error_msg:
                error_msg = str(e)
            log_error(f"Error during processing: {error_msg}")
            
            try:
                self.save_result_json(
                    dataset_name, 
                    prompt_content if 'prompt_content' in locals() else "", 
                    response if 'response' in locals() else "", 
                    None, False, temperature, metrics, error_msg
                )
            except:
                pass
            
            return False, None, None, {}

    def process_all_datasets(self, prompt_file, temperature=0.3):
        """Process all datasets in the data folder - single run only"""
        datasets = self.find_datasets()
        
        if not datasets:
            log_error("No datasets found in data folder")
            return []
        
        log_clean(f"Found {len(datasets)} datasets to process")
        
        all_results = []
        
        for dataset_idx, csv_file in enumerate(datasets, 1):
            print(f"\n{'='*80}")
            log_clean(f"DATASET {dataset_idx}/{len(datasets)}: {csv_file.name}")
            print(f"{'='*80}")
            
            success, script_path, output_path, metrics = self.process_dataset(
                csv_file, prompt_file, temperature
            )
            
            all_results.append({
                'dataset': csv_file.name,
                'success': success,
                'script_path': str(script_path) if script_path else None,
                'output_path': str(output_path) if output_path else None,
                'metrics': metrics
            })
        
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results):
        """Print summary of results"""
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ERROR DETECTION RESULTS")
        print(f"{'='*80}")
        
        total_runs = len(results)
        successful_runs = sum(1 for r in results if r['success'])
        
        print(f"Total datasets: {total_runs}")
        print(f"Successful: {successful_runs}")
        print(f"Failed: {total_runs - successful_runs}")
        print(f"Success rate: {successful_runs/total_runs*100:.1f}%")