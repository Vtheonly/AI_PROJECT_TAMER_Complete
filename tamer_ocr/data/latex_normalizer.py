"""
LaTeX Normalization Utility for Math OCR Training.

Cleans raw LaTeX labels to ensure consistency across all datasets.
This is a standalone preprocessing step that runs BEFORE tokenization.

Removes:
  - Spacing commands: \\, \\; \\! \\quad \\qquad
  - Visual-only commands: \\left \\right \\limits
  - Matrix/array environments: \\begin{array}, \\begin{matrix}

Standardizes:
  - \\over → \\frac{}{} (when possible)
  - Whitespace normalization
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("TAMER.Normalizer")

# Patterns to strip entirely (spacing and visual-only commands)
STRIP_PATTERNS = [
    (r'\\[,;!]', ''),           # \, \; \! → remove
    (r'\\quad\b', ''),          # \quad → remove
    (r'\\qquad\b', ''),         # \qquad → remove
    (r'\\left\b', ''),          # \left → remove
    (r'\\right\b', ''),         # \right → remove
    (r'\\limits\b', ''),        # \limits → remove
    (r'\\displaystyle\b', ''),  # \displaystyle → remove
    (r'\\textstyle\b', ''),     # \textstyle → remove
    (r'\\scriptstyle\b', ''),   # \scriptstyle → remove
    (r'\\mkern\s*\d+\w*', ''),  # \mkern → remove
    (r'\\mspace\s*\{[^}]*\}', ''), # \mspace → remove
    (r'\\hfill\b', ''),         # \hfill → remove
    (r'\\vfill\b', ''),         # \vfill → remove
    (r'\\phantom\s*\{[^}]*\}', ''),  # \phantom{...} → remove
]

# Patterns that indicate complex structures we should discard
DISCARD_PATTERNS = [
    r'\\begin\s*\{array\}',
    r'\\begin\s*\{matrix\}',
    r'\\begin\s*\{pmatrix\}',
    r'\\begin\s*\{bmatrix\}',
    r'\\begin\s*\{vmatrix\}',
    r'\\begin\s*\{Vmatrix\}',
    r'\\begin\s*\{cases\}',
    r'\\begin\s*\{aligned\}',
    r'\\begin\s*\{align\}',
]


def should_discard(latex: str) -> bool:
    """Check if a LaTeX string contains structures too complex for this training regime."""
    for pattern in DISCARD_PATTERNS:
        if re.search(pattern, latex):
            return True
    return False


def normalize_latex(latex: str) -> Optional[str]:
    """
    Normalize a LaTeX string for consistent training.
    
    Returns None if the sample should be discarded (too complex).
    Returns the cleaned string otherwise.
    """
    if not latex or not latex.strip():
        return None

    # Check for discard-worthy patterns first
    if should_discard(latex):
        return None

    result = latex.strip()

    # Apply all strip patterns
    for pattern, replacement in STRIP_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Standardize \over → \frac{}{} where possible
    # \over is a TeX primitive: a \over b → \frac{a}{b}
    result = _convert_over_to_frac(result)

    # Normalize whitespace: collapse multiple spaces into one
    result = re.sub(r'\s+', ' ', result).strip()

    # Remove leading/trailing braces if they wrap the entire expression
    # But be careful not to break \frac{a}{b}
    if result.startswith('{') and result.endswith('}'):
        depth = 0
        all_matched = True
        for i, c in enumerate(result):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth == 0 and i < len(result) - 1:
                all_matched = False
                break
        if all_matched:
            result = result[1:-1].strip()

    # Final check: empty string after normalization
    if not result:
        return None

    return result


def _convert_over_to_frac(latex: str) -> str:
    """
    Convert \\over usage to \\frac{}{}.
    
    TeX \\over: {a \\over b} → \\frac{a}{b}
    This is a simplified conversion that handles common cases.
    """
    if '\\over' not in latex:
        return latex

    # Pattern: {numerator \over denominator}
    # We look for the outermost braces containing \over
    result = latex
    changed = True
    max_iterations = 10  # Safety limit
    iterations = 0

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Find pattern: {content \over content}
        # Use a simple state machine approach
        i = 0
        while i < len(result):
            if result[i] == '{':
                # Find matching close brace
                depth = 1
                j = i + 1
                while j < len(result) and depth > 0:
                    if result[j] == '{':
                        depth += 1
                    elif result[j] == '}':
                        depth -= 1
                    j += 1

                # Content between braces is result[i+1:j-1]
                content = result[i+1:j-1]

                if '\\over' in content:
                    # Split on \over
                    over_idx = content.index('\\over')
                    numerator = content[:over_idx].strip()
                    denominator = content[over_idx+5:].strip()  # len('\\over') = 5

                    if numerator and denominator:
                        replacement = f'\\frac{{{numerator}}}{{{denominator}}}'
                        result = result[:i] + replacement + result[j:]
                        changed = True
                        break  # Restart search since string changed
            i += 1

    return result


def normalize_corpus(corpus: list) -> list:
    """
    Normalize an entire corpus of LaTeX strings.
    
    Args:
        corpus: List of dicts with 'latex' key
    
    Returns:
        Filtered and normalized list (discarded samples removed)
    """
    normalized = []
    discarded = 0

    for sample in corpus:
        latex = sample.get('latex', '')
        cleaned = normalize_latex(latex)

        if cleaned is None:
            discarded += 1
            continue

        new_sample = dict(sample)
        new_sample['latex'] = cleaned
        normalized.append(new_sample)

    logger.info(f"Normalized corpusx: {len(normalized)} kept, {discarded} discarded "
                f"({100*discarded/max(len(corpus),1):.1f}% filtered)")

    return normalized