"""
Code generation and validation for error detection
"""

import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from .utils import sanitize_python_code, validate_python_syntax, extract_python_code


class CodeGenerator:
    """Handles code generation and execution"""
    
    def __init__(self, api_key, model_name, script_dir):
        self.api_key = api_key
        self.model_name = model_name
        self.script_dir = Path(script_dir)
        
        # Initialize the OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            print(f"[INFO] OpenAI client initialized with model: {self.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_code(self, prompt, max_tokens=4000, temperature=0.3, max_retries=5):
        """Generate code using OpenAI API with syntax validation"""
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}: Generating detection code with model {self.model_name}...")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Python programmer specializing in data quality "
                                "and error detection for process mining event logs.\n\n"
                                "CRITICAL CODE SAFETY RULES:\n"
                                "- NEVER create multi-line f-strings without triple quotes\n"
                                "- Keep all f-strings on a single line\n"
                                "- Use simple string concatenation for complex text\n"
                                "- Always use .items() not .iteritems()\n"
                                "- All code must be syntactically valid Python\n"
                                "- Output ONLY executable Python code, no markdown\n\n"
                                "Your code will be validated before execution."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] API call failed: {error_msg}")
                
                # Check for specific error types
                if "model" in error_msg.lower() and "does not exist" in error_msg.lower():
                    print(f"[ERROR] Model '{self.model_name}' does not exist!")
                    print("[ERROR] Available models include: gpt-5.2, gpt-5.2-pro, gpt-5-mini, gpt-4-turbo, gpt-4")
                    print("[ERROR] Please use --model flag with a valid model name")
                    return None
                elif "rate limit" in error_msg.lower() and attempt < max_retries - 1:
                    wait_time = min(120, 15 * (attempt + 1))
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in 15 seconds...")
                    time.sleep(15)
                else:
                    print(f"[ERROR] Error generating code after {max_retries} attempts: {e}")
                    return None
    
    def save_generated_script(self, code, dataset_name):
        """Save the generated code to a file"""
        if not code:
            return None
        
        # Sanitize code
        code = sanitize_python_code(code)
            
        script_filename = f"{dataset_name}_detection.py"
        script_path = self.script_dir / script_filename
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(f"# Generated detection script for {dataset_name}\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Model: {self.model_name}\n\n")
            f.write(code)
        
        return script_path
    
    def run_script(self, script_path, csv_path):
        """Run the generated script using subprocess"""
        try:
            cmd = ['python', str(script_path), str(csv_path)]
            
            print(f"[DEBUG] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=os.getcwd()
            )
            
            # Print output for debugging
            if result.stdout:
                print(f"[DEBUG] Script STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"[DEBUG] Script STDERR:\n{result.stderr}")
            
            print(f"[DEBUG] Return code: {result.returncode}")
            
            return result.returncode == 0, result.stdout, result.stderr, result
            
        except subprocess.TimeoutExpired:
            print(f"[DEBUG] Script execution timed out after 600 seconds")
            return False, "", "Script execution timed out after 600 seconds", None
        except Exception as e:
            print(f"[DEBUG] Script execution error: {e}")
            return False, "", str(e), None