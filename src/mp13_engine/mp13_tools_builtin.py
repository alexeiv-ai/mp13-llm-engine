# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import re
import codecs
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, FrozenSet
import numpy as np
import numexpr as ne
import importlib

from .mp13_config import RegisteredTool
from sympy import sympify, symbols, simplify, solve, diff, integrate, SympifyError, I, expand, factor, pi, E, oo, factorial, gamma, atan2
from sympy.functions import sin, cos, tan, log, exp, sqrt, Abs, asin, acos, atan, sinh, cosh, tanh

# --- Guide Class ---
class Guide:
    """A reusable class to provide structured, searchable guidance for a tool."""
    def __init__(self, content_map: Dict[str, Union[List[str], Callable[[], List[str]]]]):
        self._content_map = content_map
        self.topics = sorted(list(content_map.keys()))

    def query(self, topic: str, search: Optional[str] = None) -> List[str]:
        """
        Retrieves and optionally filters guidance content for a given topic.
        """
        if topic not in self.topics and topic != "all":
            raise ValueError(f"Invalid topic '{topic}'. Available topics are: {self.topics}")

        topics_to_process = self.topics if topic == "all" else [topic]
        results = []
        search_lower = search.lower() if search else None

        for t in topics_to_process:
            content_source = self._content_map.get(t, [])
            items = content_source() if callable(content_source) else content_source

            if search_lower:
                filtered_items = [item for item in items if search_lower in item.lower()]
            else:
                filtered_items = items

            if filtered_items:
                results.append(f"--- {t.replace('_', ' ').title()} ---")
                results.extend(filtered_items)

        if not results:
            return ["No results found for your search criteria."]

        return results

# --- Symbolic Algebra Tool ---

_SYMBOLIC_ALGEBRA_GUIDE_CONTENT = {
    "help": [
        "This is the user manual for the 'symbolic_algebra' tool. It explains how to perform symbolic math operations.",
        "To get specific information, call this guide with a topic. For example: `symbolic_algebra_guide(topic='operations')`",
        "--- TOPICS ---",
        "* 'operations': Lists all available symbolic operations and their purpose.",
        "* 'usage': Shows practical examples with their expected output.",
        "* 'expressions': Explains how to correctly format the math expression and declare variables.",
        "* 'errors': Describes common errors and how to fix them.",
        "--- WORKFLOW ---",
        "1. **Explore:** `symbolic_algebra_guide(topic='operations')`",
        "2. **Construct & Execute:** `symbolic_algebra(expr='(x+y)**2', variables=['x', 'y'], operation='expand')`",
    ],
    "operations": [
        "The 'operation' parameter determines the action to perform. Supported operations are:",
        "* 'simplify': General-purpose simplification of an expression.",
        "* 'expand': Expands a polynomial expression. e.g., (x+y)**2 -> x**2 + 2*x*y + y**2.",
        "* 'factor': Factors a polynomial. e.g., x**2 - 4 -> (x - 2)*(x + 2).",
        "* 'solve': Solves an equation for a variable. The expression is assumed to equal zero.",
        "  - Requires 'wrt_variable' to specify which variable to solve for.",
        "* 'diff': Differentiates an expression with respect to a variable.",
        "  - Requires 'wrt_variable'.",
        "* 'integrate': Integrates an expression with respect to a variable (indefinite integral).",
        "  - Requires 'wrt_variable'.",
    ],
    "usage": [
        "The tool returns the result of the symbolic operation as a single string.",
        "---",
        "**Example 1: Simplifying an expression**",
        "  Call: `symbolic_algebra(expr='sin(x)**2 + cos(x)**2', variables=['x'], operation='simplify')`",
        "  Returns: `'1'`",
        "---",
        "**Example 2: Expanding a polynomial**",
        "  Call: `symbolic_algebra(expr='(a+b)**2', variables=['a', 'b'], operation='expand')`",
        "  Returns: `'a**2 + 2*a*b + b**2'`",
        "---",
        "**Example 3: Solving an equation for a variable**",
        "  The expression is assumed to be equal to zero.",
        "  Call: `symbolic_algebra(expr='x**2 - 4', variables=['x'], operation='solve', wrt_variable='x')`",
        "  Returns: `'[-2, 2]'`",
        "---",
        "**Example 4: Differentiating an expression**",
        "  Call: `symbolic_algebra(expr='x**3 * sin(x)', variables=['x'], operation='diff', wrt_variable='x')`",
        "  Returns: `'x**3*cos(x) + 3*x**2*sin(x)'`",
        "---",
        "**Example 5: Integrating an expression**",
        "  Call: `symbolic_algebra(expr='cos(x)', variables=['x'], operation='integrate', wrt_variable='x')`",
        "  Returns: `'sin(x)'`",
    ],
    "expressions": [
        "The 'expr' field must contain a single, valid mathematical expression as a string.",
        "Assignments (e.g., 'y = x**2') and multi-statement scripts are NOT allowed.",
        "The 'variables' field must be a list of strings, containing ALL symbols present in the expression.",
        "Example: For 'a*x**2 + b*x + c', `variables` must be `['a', 'x', 'b', 'c']`.",
        "--- Supported Constants ---",
        "pi, e (Euler's number), I (imaginary unit), oo (infinity)",
        "--- Supported Functions ---",
        "Trigonometric: sin, cos, tan, asin, acos, atan, atan2",
        "Hyperbolic: sinh, cosh, tanh",
        "Exponential/Logarithmic: exp, log",
        "Other: sqrt, Abs (absolute value), factorial, gamma",
    ],
    "errors": [
        "ValueError: The most common error. It can be caused by:",
        "  - Using an unknown operation.",
        "  - Forgetting to provide 'wrt_variable' for solve, diff, or integrate.",
        "  - Forgetting to declare a symbol in the 'variables' list. The tool will tell you which symbols are missing.",
        "  - Providing an invalid expression (e.g., with assignments or multiple statements).",
        "SympifyError: Occurs if the expression string has a syntax error that SymPy cannot parse.",
    ],
}
_symbolic_algebra_guide = Guide(_SYMBOLIC_ALGEBRA_GUIDE_CONTENT)

def symbolic_algebra(expr: str, variables: List[str], operation: str = "simplify", wrt_variable: Optional[str] = None, **kwargs):
    """
    Performs symbolic algebraic manipulations on mathematical expressions.

    This tool uses SymPy to work with mathematical expressions as symbols, allowing for
    exact, non-numerical answers. It is suitable for algebra and calculus tasks.

    Args:
        expr: The mathematical expression as a string.
        variables: A list of ALL symbolic variable names in the expression.
        operation: The operation to perform. One of 'simplify', 'solve', 'diff', 'integrate', 'expand', 'factor'.
        wrt_variable: The variable to operate on for 'solve', 'diff', or 'integrate'.

    Returns:
        The result of the operation as a string.

    Raises:
        ValueError: If the operation is invalid, a variable is missing, or the expression is malformed.
    """
    tool_call = kwargs.get('tool_call')
    try:
        # Sanitize the expression. Symbolic operations expect a single expression,
        # not statements with imports or assignments.
        sanitized_expr = _sanitize_expression(expr, allow_statements=False)
    except ValueError as e:
        error_str = tool_call.normalize_error(e) if tool_call else str(e)
        raise ValueError(f"Invalid expression format for symbolic algebra: {error_str}. Tip: Use the `symbolic_algebra_guide` tool for help.")

    allowed_ops = {"simplify", "solve", "diff", "integrate", "expand", "factor"}
    if operation not in allowed_ops:
        raise ValueError(f"Operation must be one of {allowed_ops}")

    # Define the symbols for the variables
    sym_vars = {v: symbols(v) for v in variables}
    
    # Define a safe namespace for parsing the expression
    # This includes the declared variables and standard sympy functions
    safe_namespace = {
        **sym_vars,
        # Constants
        'pi': pi, 'e': E, 'I': I, 'oo': oo,
        # Functions
        'sin': sin, 'cos': cos, 'tan': tan,
        'asin': asin, 'acos': acos, 'atan': atan, 'atan2': atan2,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'log': log, 'exp': exp, 'sqrt': sqrt, 'Abs': Abs,
        'factorial': factorial, 'gamma': gamma,
    }

    # --- Pre-validation for undeclared variables ---
    # Find all identifiers in the expression and check if they are either in the safe namespace or declared variables.
    all_identifiers = set(_ID_RE.findall(sanitized_expr))
    known_symbols = set(safe_namespace.keys())
    undeclared = all_identifiers - known_symbols
    if undeclared:
        raise ValueError(f"The following symbols were used in the expression but not declared in the 'variables' list: {sorted(list(undeclared))}")
        
    try:
        # Safely parse the string expression
        sympy_expr = sympify(sanitized_expr, locals=safe_namespace)
    except (SympifyError, TypeError) as e:
        error_str = tool_call.normalize_error(e) if tool_call else str(e)
        raise ValueError(f"Failed to parse expression '{sanitized_expr}': {error_str}")

    result = ""
    # Perform the requested operation
    if operation == "simplify":
        result = simplify(sympy_expr)
    elif operation == "expand":
        result = expand(sympy_expr)
    elif operation == "factor":
        result = factor(sympy_expr)
    elif operation in ["solve", "diff", "integrate"]:
        if not wrt_variable:
            raise ValueError(f"The 'wrt_variable' parameter is required for the '{operation}' operation.")
        if wrt_variable not in sym_vars:
            raise ValueError(f"The 'wrt_variable' '{wrt_variable}' was not declared in the 'variables' list.")
        
        target_var = sym_vars[wrt_variable]
        if operation == "solve":
            result = solve(sympy_expr, target_var)
        elif operation == "diff":
            result = diff(sympy_expr, target_var)
        elif operation == "integrate":
            result = integrate(sympy_expr, target_var)
    else:
        # This case should not be reached due to the initial check
        raise NotImplementedError(f"Operation '{operation}' is not implemented.")

    return str(result)

def symbolic_algebra_guide(topic: str, search: Optional[str] = None) -> List[str]:
    """Provides detailed, searchable guidance on using the symbolic_algebra tool."""
    return _symbolic_algebra_guide.query(topic, search)

# --- Scriptable Calculator Tool ---

_SCRIPTABLE_CALCULATOR_GUIDE_CONTENT = {
    "help": [
        "This is the user manual for the 'scriptable_calculator' tool. It explains how to use the tool correctly.",
        "To get specific information, call this guide with a topic. For example: `scriptable_calculator_guide(topic='usage')`",
        "--- TOPICS ---",
        "* 'usage': Shows practical examples with their expected output.",
        "* 'syntax': Explains the rules for writing expressions.",
        "* 'allowed_symbols': Lists all available math functions and constants.",
        "* 'errors': Describes common errors and how to fix them.",
        "--- WORKFLOW ---",
        "1. **Explore:** `scriptable_calculator_guide(topic='usage')`",
        "2. **Construct & Execute:** `scriptable_calculator(expr='r=5; area=pi*r**2; area')`",
    ],
    "syntax": [
        "Statements can be separated by semicolons (;) or newlines.",
        "Statements before the final one can be assignments (e.g., 'x=5') or expressions (e.g., '2*pi'); the results of intermediate expressions are discarded.",
        "The last statement is the result. It can be a single numerical expression, an assignment, or a tuple of variables.",
        "A tuple of variables (e.g., 'a, b' or '(a, b)') is useful for ensuring specific variables are included in the output dictionary.",
        "Comments are not supported.",
        "Complex control flow (loops, if/else) is not supported.",
        "Dictionary literals (e.g., {'key': value}) are not supported as expressions.",
    ],
    "usage": [
        "The tool returns a dictionary. The key 'result' holds the value of the final expression. Other keys hold the values of assigned variables.",
        "---",
        "**Example 1: Simple evaluation**",
        "  Call: `scriptable_calculator(expr='2 * pi * 5')`",
        "  Returns: `{'result': 31.4159...}`",
        "---",
        "**Example 2: Multi-step calculation**",
        "  The last statement is an expression ('area'), so its value is put into 'result'.",
        "  Call: `scriptable_calculator(expr='r=5; area=pi*r**2; area')`",
        "  Returns: `{'result': 78.5398...}`",
        "---",
        "**Example 3: Using `return_assignments='all'`**",
        "  This returns all variables that were assigned a value.",
        "  Call: `scriptable_calculator(expr='r=5; area=pi*r**2; area', return_assignments='all')`",
        "  Returns: `{'r': 5, 'area': 78.5398..., 'result': 78.5398...}`",
        "---",
        "**Example 4: Returning a tuple of variables**",
        "  The last statement is a tuple of variable names, which forces them into the output.",
        "  Call: `scriptable_calculator(expr='x=10; y=20; z=x+y; x,y')`",
        "  Returns: `{'x': 10, 'y': 20}`",
        "---",
        "**Common Mistake: Using Dictionary Syntax**",
        "  The tool does not support Python dictionary literals as the final expression.",
        "  INCORRECT: `scriptable_calculator(expr=\"{'circle_area': 34.2, 'triangle_area': 7.5}\")`",
        "  CORRECT WAY: Assign values to variables and return them as a tuple.",
        "  `expr='circle_area=34.2; triangle_area=7.5; circle_area, triangle_area'`",
    ],
    "limitations": [
        "This tool is for numerical evaluation, not symbolic math. For symbolic operations, use 'symbolic_algebra'.",
        "Variables passed in the 'variables' argument must be scalars (single numbers), not arrays or lists.",
        "The script must be self-contained. It cannot call external libraries or access the file system.",
        "Lambda functions and Python function definitions ('def') are not supported.",
    ],
    "allowed_symbols": lambda: sorted(list(ALLOWED_FUNCTIONS_AND_CONSTANTS) + ['radians', 'degrees']),
    "errors": [
        "KeyError: Occurs if you use a variable that has not been defined in the script or passed in the 'variables' dict. e.g., 'a + b' without defining 'a' or 'b'.",
        "ValueError: Occurs for illegal tokens (e.g., trying to use a disallowed function like 'eval') or syntax errors.",
        "AttributeError: Often occurs with invalid statements that are not simple 'var = expr' assignments.",
        "TypeError: Can occur if the final expression is not a valid numerical formula. Note that tuples are only supported if they contain only variable names (e.g., 'a, b'), not expressions (e.g., 'a+1, b').",
    ],
}
_scriptable_calculator_guide = Guide(_SCRIPTABLE_CALCULATOR_GUIDE_CONTENT)

def scriptable_calculator_guide(topic: str, search: Optional[str] = None) -> List[str]:
    """Provides detailed, searchable guidance on using the scriptable_calculator tool."""
    return _scriptable_calculator_guide.query(topic, search)

#: Constants that must be manually added to the numexpr environment because
#: providing a `local_dict` bypasses the compiler's special handling for them.
_NUMEXPR_MANUAL_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
    "True": True,
    "False": False,
    "None": None,
}
# ------------------------------------------------------------------ helpers
_ID_RE = re.compile(r"[A-Za-z_]\w*")          # Python identifier pattern


def _discover_allowed() -> FrozenSet[str]:
    """Runtime snapshot of every legal NumExpr symbol (functions + constants)."""
    funcs = set(getattr(ne.expressions, "functions", {}).keys())

    # constants table moved in 2.11
    if hasattr(ne.expressions, "supported_constants"):
        consts = set(ne.expressions.supported_constants)
    else:                                      # ≥ 2.11.0 fallback
        nec = importlib.import_module("numexpr.necompiler")
        consts = set(getattr(nec, "scalar_constant_kinds", ()))

    # Add our manually managed constants to the allowed set.
    consts |= set(_NUMEXPR_MANUAL_CONSTANTS.keys())
    return frozenset(x for x in funcs | consts if not x.startswith("_"))


ALLOWED_FUNCTIONS_AND_CONSTANTS: FrozenSet[str] = _discover_allowed()
# ------------------------------------------------------------------ validation

# Matches standalone import/from lines without consuming following statements.
_SANITIZE_IMPORTS_RE = re.compile(
    r"(?im)^[ \t]*(?:from[ \t]+[\w\.]+[ \t]+import[ \t]+[^\n;]+|import[ \t]+[^\n;]+)[ \t]*;?[ \t]*$"
)
_SANITIZE_ATTR_RE = re.compile(r"\b[a-zA-Z_]\w*\.(\w+)\b")
def _sanitize_expression(src: str, allow_statements: bool = True) -> str:
    """
    Sanitizes an expression string by removing comments, imports, and module qualifications.
    Optionally disallows multi-statement expressions.
    """
    # 1. Remove comments
    sanitized = re.sub(r'#.*$', '', src, flags=re.MULTILINE)

    # 2. Remove import statements
    sanitized = _SANITIZE_IMPORTS_RE.sub("", sanitized)

    # 3. Remove module qualifications (e.g., "math.sqrt" -> "sqrt")
    sanitized = _SANITIZE_ATTR_RE.sub(r"\1", sanitized)

    # 4. Check for disallowed characters/statements if needed
    if not allow_statements:
        if ';' in sanitized:
            raise ValueError("Multi-statement expressions (using ';') are not allowed for this operation.")
        # Check for assignment operator '=' that is not part of a comparison operator '==' '!=' '>=' '<='
        if re.search(r'(?<![<>!=])=(?![=])', sanitized):
            raise ValueError("Assignment statements (using '=') are not allowed for this operation.")

    return sanitized.strip()
# ------------------------------------------------------------------ public API

def _convert_numpy_to_python(data: Any) -> Any:
    """Recursively converts numpy types in a data structure to native Python types."""
    if isinstance(data, dict):
        return {k: _convert_numpy_to_python(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_convert_numpy_to_python(i) for i in data)
    if isinstance(data, np.ndarray):
        # Convert 0-dim array to a scalar, otherwise to a list.
        return data.item() if data.ndim == 0 else data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    return data

def _validate_tokens(tokens: List[str], env: Dict[str, Any]):
    """Check for illegal tokens not in the environment or NumExpr's allowlist."""
    bad_tokens = {t for t in tokens if t not in env and t not in ALLOWED_FUNCTIONS_AND_CONSTANTS}
    if bad_tokens:
        raise ValueError(f"Illegal token(s): {', '.join(sorted(bad_tokens))}")


def scriptable_calculator(expr: Optional[str] = None, *, variables: Optional[Dict[str, Any]] = None,
                       return_assignments: str = "outputs", **kwargs) -> Dict[str, Any]: # type: ignore
    """
    A scriptable calculator that evaluates mathematical expressions and assignments.
    If the script ends with a lone `outputs` or `all` token, it returns all assignments.

    Parameters
    ----------
    expr : str, optional
        Semicolon- or newline-separated statements. The *last* element may
        be an assignment or a pure expression. NumExpr still sees only one
        expression at a time.
    variables : dict[str, Any], optional
        External constants (scalars only, cannot be arrays). Can be used to override built-in constants.
    return_assignments : {"outputs", "all"}, optional
        * "outputs":return only variables that are never referenced by
          later statements.
        * "all": return every assignment made.
    **kwargs : dict, optional
        If the tool call arguments were malformed, this dictionary contains the
        raw, unparsed payload under the key 'tool_args_issue' (e.g., {'_non_parsed': '...'} or {'_string_value': '...'}).
        The function will attempt to recover the 'expr' from this payload.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the requested assigned variables and/or the final expression result.
    """
    warning_message = None
    recovery_used = False
    tool_args_issue = kwargs.get('tool_args_issue')
    tool_call = kwargs.get('tool_call')
    unescape_error = None

    def _find_unescaped_quote(text: str, start_index: int) -> int:
        esc = False
        for i in range(start_index, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                return i
        return -1

    if not expr and tool_args_issue:
        # Standard recovery path for malformed arguments.
        raw_val = tool_args_issue.get('_non_parsed') or tool_args_issue.get('_string_value')
        recovered_expr = None
        if isinstance(raw_val, str):
            try:
                # First, unescape the string to handle `\\"` and `\\n`.
                unescaped_val = codecs.decode(raw_val, 'unicode_escape')
                
                # New, simpler recovery: find the 'expr' value directly.
                expr_marker = '"expr": "'
                start_index = unescaped_val.rfind(expr_marker)
                
                if start_index != -1:
                    start_index += len(expr_marker)
                    # The rest of the string is the expression, possibly with trailing junk.
                    end_index = _find_unescaped_quote(unescaped_val, start_index)
                    if end_index > start_index:
                        recovered_expr = unescaped_val[start_index:end_index]
                    else:
                        # No closing quote, take the rest of the string and clean trailing junk.
                        recovered_expr = unescaped_val[start_index:].rstrip('"}')
                else: 
                    # Fallback if marker is not found
                    recovered_expr = unescaped_val
            except Exception as e:
                recovered_expr = raw_val
                unescape_error = f"recovery_failed: {e}"

        elif isinstance(raw_val, dict):
             recovered_expr = raw_val.get('expr')
        
        if recovered_expr:
            expr = recovered_expr
            recovery_used = True
            warning_message = f"Tool call arguments format had problems, used best guessed expr value: '{expr}'"
            if tool_call and 'KeepRaw' not in tool_call.action:
                tool_call.action.append('KeepRaw')


    if not expr:
        raise ValueError("The 'expr' argument is required and could not be recovered from the malformed tool call.")

    try:
        variables = variables or {}

        # ---------- first pass: dependency graph ------------------------
        expr = _sanitize_expression(expr, allow_statements=True)
        stmts = [s.strip() for s in re.split(r"[;\n]+", expr) if s.strip()]
        if not stmts:
            raise ValueError("empty script")

        # Allow a trailing sentinel 'outputs' / 'all' to request all assignments.
        last_stmt_lower = stmts[-1].lower()
        if last_stmt_lower in {"outputs", "all"}:
            stmts.pop()
            return_assignments = "all"
            if not stmts:
                raise ValueError("No calculable statements provided before 'outputs'.")

        if warning_message and stmts and stmts[-1].lower() == 'outputs':
            stmts.pop()
            return_assignments = 'all'

        assigned, used = [], set()
        for i, s in enumerate(stmts):
            m = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)", s)
            if m:
                lhs, rhs = m.groups()
                assigned.append(lhs)
                used.update(_ID_RE.findall(rhs))
            else:
                # If this is the last statement and it's a tuple of variables,
                # don't add them to the 'used' set, so they appear in the output dict.
                is_last_stmt = (i == len(stmts) - 1)
                is_tuple_of_vars_for_dep = False
                if is_last_stmt and ',' in s:
                    s_stripped = s.strip()
                    if s_stripped.startswith('(') and s_stripped.endswith(')'):
                        s_stripped = s_stripped[1:-1].strip()
                    potential_vars = [p.strip() for p in s_stripped.split(',')]
                    if all(potential_vars) and all(_ID_RE.fullmatch(p) for p in potential_vars):
                        is_tuple_of_vars_for_dep = True

                if not is_tuple_of_vars_for_dep:
                    used.update(_ID_RE.findall(s))

        outputs = {v for v in assigned if v not in used}

        # ---------- execution loop --------------------------------------
        env = {}
        # Populate the environment with helper functions and all manually managed constants.
        # When a local_dict is provided, numexpr doesn't use its global context.
        # so we must provide everything.
        env.update(_NUMEXPR_MANUAL_CONSTANTS)
        env.update({"radians": np.radians, "degrees": np.degrees})
        env.update(variables)  # User-provided variables take precedence
        for s in stmts[:-1]:
            if s.lstrip().startswith("def "):
                raise ValueError("Function definitions ('def') are not supported. The script should only contain numerical expressions and simple assignments.")
            match = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)", s)
            if match:
                # It's an assignment, process it.
                lhs, rhs = match.groups()
                _validate_tokens(_ID_RE.findall(rhs), env)
                env[lhs] = ne.evaluate(rhs, local_dict=env)
            else:
                # It's not an assignment. Treat as a "scratchpad" calculation.
                # Validate its tokens and evaluate it, but discard the result.
                _validate_tokens(_ID_RE.findall(s), env)
                ne.evaluate(s, local_dict=env)

        last = stmts[-1]
        m = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)", last)
        result_key = None

        # --- Tuple detection logic ---
        is_tuple_of_vars = False
        tuple_vars = []
        # Check if it looks like a tuple of plain variables (e.g., "a, b" or "(a, b)")
        if ',' in last:
            s_stripped = last.strip()
            if s_stripped.startswith('(') and s_stripped.endswith(')'):
                s_stripped = s_stripped[1:-1].strip()
            
            potential_vars = [p.strip() for p in s_stripped.split(',')]
            if all(potential_vars) and all(_ID_RE.fullmatch(p) for p in potential_vars):
                is_tuple_of_vars = True
                tuple_vars = potential_vars

        if m:                                     # trailing assignment
            lhs, rhs = m.groups()
            _validate_tokens(_ID_RE.findall(rhs), env)
            env[lhs] = ne.evaluate(rhs, local_dict=env)
            result = None  # No explicit final expression.
            outputs.add(lhs)
        elif is_tuple_of_vars:
            result = None # No primary result.
            _validate_tokens(tuple_vars, env) # Validate the variables exist.
        else:                                     # trailing pure expression
            if last.strip().startswith('{'):
                raise ValueError(
                    "Dictionary literals are not supported as the final expression. "
                    "To return multiple values, assign them to variables and list the variable names as a tuple in the final statement. "
                    "Example: `area_circle=34; area_triangle=7.5; area_circle, area_triangle`"
                )
            _validate_tokens(_ID_RE.findall(last), env)
            result = ne.evaluate(last, local_dict=env)
            result_key = "result" if "result" not in env else "output"
            env[result_key] = result
            outputs.add(result_key)

        # ---------- what to expose? -------------------------------------
        if return_assignments not in {"outputs", "all"}:
            raise ValueError("return_assignments must be 'outputs' or 'all'")

        if return_assignments == "all":
            keys_to_include = set(assigned)
            if result_key:
                keys_to_include.add(result_key)
        else:  # "outputs"
            keys_to_include = outputs

        final_dict = {k: env[k] for k in keys_to_include if k in env}
        
        if unescape_error:
            final_dict.setdefault('warning', '')
            final_dict['warning'] = (final_dict['warning'] + ' ' + unescape_error).strip()
        
        if warning_message:
            final_dict.setdefault('warning', '')
            final_dict['warning'] = (final_dict['warning'] + ' ' + warning_message).strip()

        return _convert_numpy_to_python(final_dict)
    except (ValueError, KeyError, TypeError, NameError, SyntaxError) as e:
        tool_call = kwargs.get('tool_call')
        if recovery_used:
            error_message = (
                "Error during calculation: malformed or truncated tool call arguments. "
                "Please resend the tool call with a complete 'expr' string. "
                "Tip: Use the `scriptable_calculator_guide` tool for help on syntax and functions."
            )
            raise type(e)(error_message) from e
        error_str = tool_call.normalize_error(e) if tool_call else str(e)
        error_message = (
            f"Error during calculation: {error_str}. "
            "Tip: Use the `scriptable_calculator_guide` tool for help on syntax and functions. "
            "Example: `scriptable_calculator_guide(topic='help')`"
        )
        raise type(e)(error_message) from e

# --- Tool Definitions and Registration ---

symbolic_algebra_def = {
    "type": "function",
    "function": {
        "name": "symbolic_algebra",
        "description": "Performs symbolic algebraic manipulations on mathematical expressions, such as simplifying, expanding, factoring, solving equations, and calculus operations (differentiation, integration). This tool works with symbols, not numerical values.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "The mathematical expression as a string. The expression is assumed to be equal to zero for 'solve' operations."},
                "variables": {"type": "array", "items": {"type": "string"}, "description": "A list of ALL symbolic variable names in the expression (e.g., ['x', 'a', 'b', 'c'])."},
                "operation": {
                    "type": "string",
                    "description": "The operation to perform. One of 'simplify', 'solve', 'diff', 'integrate', 'expand', 'factor'.",
                    "enum": ["simplify", "solve", "diff", "integrate", "expand", "factor"]
                },
                "wrt_variable": {
                    "type": "string",
                    "description": "The variable to solve for, differentiate, or integrate with respect to. Required for 'solve', 'diff', and 'integrate' operations."
                }
            },
            "required": ["expr", "variables", "operation"]
        }
    }
}

symbolic_algebra_guide_def = {
    "type": "function",
    "function": {
        "name": "symbolic_algebra_guide",
        "description": "Provides detailed guidance on using the symbolic_algebra tool. Use topic='help' to see all topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The guidance topic to retrieve.",
                    "enum": ["help", "operations", "expressions", "usage", "errors", "all"],
                },
                "search": { "type": "string", "description": "An optional substring to filter the results. Case-insensitive." }
            },
            "required": ["topic"]
        }
    }
}

scriptable_calculator_def = {
    "type": "function",
    "function": {
        "name": "scriptable_calculator",
        "description": "A scriptable calculator that evaluates mathematical expressions and assignments. It uses the NumExpr library for safe, fast numerical computation. Ideal for multi-step calculations where intermediate results are stored in variables.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "Semicolon- or newline-separated statements. If the last statement is an expression, its value is returned as the primary result."},
                "variables": {"type": "object", "description": "A dictionary mapping external variable names to their numerical values. Can be used to override built-in constants like 'pi'."},
                "return_assignments": {
                    "type": "string",
                    "description": "Controls which assigned variables are returned. 'outputs' returns only variables not used later. 'all' returns all assignments.",
                    "enum": ["outputs", "all"],
                    "default": "outputs"
                }
            },
            "required": ["expr"]
        }
    }
}

scriptable_calculator_guide_def = {
    "type": "function",
    "function": {
        "name": "scriptable_calculator_guide",
        "description": "Provides detailed guidance on using the scriptable_calculator tool. Use topic='help' to see all topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The guidance topic to retrieve.",
                    "enum": ["help", "syntax", "usage", "limitations", "allowed_symbols", "errors", "all"]
                },
                "search": { "type": "string", "description": "An optional substring to filter the results. Case-insensitive." }
            },
            "required": ["topic"]
        }
    }
}


INTRINSICS_REGISTRY: Dict[str, RegisteredTool] = {
    "symbolic_algebra": RegisteredTool(
        name="symbolic_algebra",
        definition=symbolic_algebra_def,
        implementation=symbolic_algebra,
        guide_definition=symbolic_algebra_guide_def,
        guide_implementation=symbolic_algebra_guide,
        is_intrinsic=True
    ),
    "scriptable_calculator": RegisteredTool(
        name="scriptable_calculator",
        definition=scriptable_calculator_def,
        implementation=scriptable_calculator,
        guide_definition=scriptable_calculator_guide_def,
        guide_implementation=scriptable_calculator_guide,
        is_intrinsic=True
    ),
}
