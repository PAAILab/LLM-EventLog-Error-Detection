#!/usr/bin/env python3
"""
GPT-based Error Detection Processor - Main Entry Point
Processes datasets from data folder using prompts as-is from prompt folder
"""

import os
import argparse
from pathlib import Path
from workflow.detector import GPTDetectorProcessor


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GPT-based Error Detection Processor - Single Run")
    
    parser.add_argument("--prompt-file", default="prompt/vanilla_examples.txt", 
                       help="Path to the prompt file")
    parser.add_argument("--api-key", help="OpenAI API token")
    parser.add_argument("--temperature", type=float, default=0.3, 
                       help="Temperature for generation")
    parser.add_argument("--model", type=str, default=None, 
                       help="GPT model to use (overrides OPENAI_MODEL env var)")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing datasets")
    
    args = parser.parse_args()
    
    # Read model from env variable first, then command line, then default
    if args.model:
        model = args.model
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    
    print("[INFO] Starting GPT-based Error Detection Pipeline")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Temperature: {args.temperature}")
    if args.model:
        print(f"[INFO] Model source: command line argument")
    elif "OPENAI_MODEL" in os.environ:
        print(f"[INFO] Model source: OPENAI_MODEL environment variable")
    else:
        print(f"[INFO] Model source: default (gpt-5.2)")
    print(f"[INFO] To use a different model:")
    print(f"[INFO]   - Set OPENAI_MODEL environment variable: export OPENAI_MODEL=gpt-5-mini")
    print(f"[INFO]   - Or use --model flag: python script.py --model gpt-5.2-pro")
    
    try:
        processor = GPTDetectorProcessor(api_key=args.api_key, model_name=model)
        
        if args.data_dir != "data":
            processor.data_dir = Path(args.data_dir)
        
        results = processor.process_all_datasets(args.prompt_file, args.temperature)
        
        print(f"\n[INFO] Processing complete!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()