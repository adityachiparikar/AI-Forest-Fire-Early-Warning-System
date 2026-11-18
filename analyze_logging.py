import ast
import os
import sys

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {
            'file': filepath,
            'error': f'Syntax error: {e}'
        }

    # Track imports
    imports = []  # list of (alias, node) for logging imports
    from_imports = []  # list of (module, names, alias) for from logging import ...

    # Track logger assignments: variable -> call node
    loggers = {}  # variable name -> assignment node

    # Track usage: set of names used (for logging module and logger variables)
    used_names = set()

    for node in ast.walk(tree):
        # Import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'logging':
                    imports.append((alias.asname or alias.name, node))
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'logging':
                # Import specific names from logging
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    from_imports.append((node.module, name, asname, node))

        # Assignments: look for logger = logging.getLogger(...)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check if the value is a call to logging.getLogger
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Attribute):
                            if node.value.func.attr == 'getLogger':
                                # Check if the value is logging or a logger from logging import
                                # We'll need to resolve the function, but for simplicity, we'll assume it's logging.getLogger
                                # We'll store the variable name and later check if it's used
                                loggers[target.id] = node.value

        # Usage: attribute access (e.g., logging.info, logger.info)
        elif isinstance(node, ast.Attribute):
            # We'll record the attribute name (e.g., 'info') and the value (which could be logging or logger)
            # We'll store the full attribute chain? We'll just store the attribute name for simplicity.
            # We'll also record the value if it's a Name (to know if it's logging or a logger variable)
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
            # Also, if the value is an attribute, we ignore for now (too complex)

        # Also, we can check for calls to logging.getLogger (but we already caught assignments)
        # We'll also check for direct calls like logging.debug, etc.
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    used_names.add(node.func.value.id)

    # Now, determine unused imports
    unused_imports = []
    for alias, node in imports:
        if alias not in used_names:
            unused_imports.append(('import', alias, node))

    unused_from_imports = []
    for module, name, asname, node in from_imports:
        if asname not in used_names:
            unused_from_imports.append(('from', module, name, asname, node))

    # Determine unused loggers
    unused_loggers = []
    for var_name, assign_node in loggers.items():
        if var_name not in used_names:
            unused_loggers.append((var_name, assign_node))

    # Suggestions for simplification: if a logger is used only once, consider using logging directly?
    # We'll skip for now.

    return {
        'file': filepath,
        'unused_imports': unused_imports,
        'unused_from_imports': unused_from_imports,
        'unused_loggers': unused_loggers,
        'used_names': used_names,
        'loggers': loggers
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logging.py <file1> [file2 ...]")
        sys.exit(1)

    files = sys.argv[1:]
    results = []
    for f in files:
        if os.path.exists(f):
            results.append(analyze_file(f))
        else:
            print(f"Warning: File {f} not found", file=sys.stderr)

    # Print summary
    for res in results:
        if 'error' in res:
            print(f"{res['file']}: {res['error']}")
            continue

        print(f"\n=== {res['file']} ===")
        if res['unused_imports']:
            print("  Unused imports:")
            for imp in res['unused_imports']:
                print(f"    {imp[0]} {imp[1]} (line {imp[2].lineno})")
        if res['unused_from_imports']:
            print("  Unused from imports:")
            for imp in res['unused_from_imports']:
                print(f"    from {imp[1]} import {imp[2]} as {imp[3]} (line {imp[4].lineno})")
        if res['unused_loggers']:
            print("  Unused loggers:")
            for logger in res['unused_loggers']:
                print(f"    {logger[0]} = logging.getLogger(...) (line {logger[1].lineno})")

        # Also show used names for debugging
        if res['used_names']:
            print(f"  Used names: {', '.join(sorted(res['used_names']))}")
        if res['loggers']:
            print(f"  Loggers defined: {', '.join(res['loggers'].keys())}")

if __name__ == '__main__':
    main()