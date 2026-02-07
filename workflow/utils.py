"""
Utility functions for error detection workflow
"""

import re


def sanitize_python_code(code: str) -> str:
    """
    Remove markdown fences and non-code artifacts from LLM output.
    Guarantees raw executable Python.
    """
    if not code:
        return code

    # Remove ALL occurrences of markdown fences (case-insensitive, all variants)
    code = re.sub(r'```\s*python\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'```\s*py\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'```', '', code)

    # Split into lines for per-line cleaning
    lines = code.splitlines()
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just markdown artifacts
        if stripped in ('', 'python', 'py', '```', '```python', '```py'):
            continue
        # Skip lines that start with markdown fence indicators
        if stripped.startswith(('```', '# ```')):
            continue
        cleaned_lines.append(line)

    cleaned_code = "\n".join(cleaned_lines).strip()

    # Final sanity check
    if cleaned_code and not cleaned_code.startswith(("import", "from", "def", "class", "#")):
        print(f"[WARNING] Generated code does not start with a typical Python statement")
        print(f"[WARNING] First line: {cleaned_code.split(chr(10))[0][:100]}")

    return cleaned_code


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python code syntax by attempting to compile it.
    Returns (is_valid, error_message)
    """
    try:
        compile(code, '<generated>', 'exec')
        return True, ""
    except SyntaxError as e:
        error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n  {e.text.strip()}"
        return False, error_msg
    except Exception as e:
        return False, str(e)


def extract_python_code(response_text):
    """Extract Python code from the response"""
    if not response_text:
        return None
        
    # Look for code blocks with python specification
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # Look for code blocks without language specification
    code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # Look for code blocks with just ```
    code_blocks = re.findall(r'```(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        lines = code.split('\n')
        if lines and lines[0].strip() in ['python', 'py']:
            code = '\n'.join(lines[1:])
        return code
    
    return response_text


def log_clean(message):
    """Clean logging to console"""
    print(f"[INFO] {message}")


def log_error(message):
    """Error logging to console"""
    print(f"[ERROR] {message}")