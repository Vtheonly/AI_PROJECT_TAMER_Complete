"""
LaTeX Normalization Utility for Math OCR Training.

Cleans raw LaTeX labels to ensure consistency across all datasets.
This is a standalone preprocessing step that runs BEFORE tokenization.

v2.3 Changes:
  - REMOVED matrix/aligned/cases from DISCARD_PATTERNS.
    These are now KEPT and the model learns to produce them.
    This is critical for multi-line equation and matrix support.

  - Added complexity classification (simple/medium/complex) for
    curriculum learning. The trainer can progressively introduce
    harder samples.

  - Only truly unparseable content is discarded now:
    tikzpicture, figure, includegraphics, etc.

Removes:
  - Spacing commands: \\, \\; \\! \\quad \\qquad
  - Visual-only commands: \\left \\right \\limits

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

# ----------------------------------------------------------------
# DISCARD_PATTERNS — Only truly unparseable content
#
# IMPORTANT: Matrices, aligned, cases, arrays are NOT discarded.
# They are kept for multi-line equation training.
# ----------------------------------------------------------------
DISCARD_PATTERNS = [
    r'\\begin\s*\{tikzpicture\}',    # vector graphics — can't OCR these
    r'\\begin\s*\{figure\}',          # figure environments
    r'\\begin\s*\{tabular\}',         # tables (not math)
    r'\\begin\s*\{table\}',           # table floats
    r'\\includegraphics',             # image references
    r'\\usepackage',                  # preamble commands
    r'\\documentclass',               # preamble commands
    r'\\newcommand',                  # macro definitions
    r'\\def\\',                       # TeX macro definitions
]

# ----------------------------------------------------------------
# Complexity Classification for Curriculum Learning
#
# The trainer uses these to progressively introduce harder samples:
#   - Epochs 1-15:  SIMPLE only (single-line formulas)
#   - Epochs 16-35: + MEDIUM (aligned, cases — multi-line)
#   - Epochs 36+:   + COMPLEX (matrices, arrays — 2D grids)
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Complexity Classification for Curriculum Learning
# ----------------------------------------------------------------

def get_complexity(latex: str) -> str:
    """
    Classify a LaTeX string by calculating a structural difficulty score.
    """
    if not latex:
        return 'simple'

    # 1. IMMEDIATE COMPLEX TRIGGERS (2D Grids, Matrices, Multi-line)
    # Any of these instantly makes it 'complex' because it breaks 1D left-to-right reading.
    complex_keywords = [
        'matrix', 'array', 'aligned', 'align', 'cases', 
        'gathered', 'split', 'eqnarray', 'multline', 
        '\\\\', '&'
    ]
    if any(kw in latex for kw in complex_keywords):
        return 'complex'

    # 2. CALCULATE STRUCTURAL SCORE
    score = 0.0
    
    # A. Length penalty: Longer sequences stress the attention matrix
    # +1 point for every 25 characters
    score += len(latex) / 25.0

    # B. Spatial shifts: Up/Down reading
    # +1 point for every superscript or subscript
    score += latex.count('^') * 1.0
    score += latex.count('_') * 1.0

    # C. Vertical / Heavy mathematical structures
    # +2 points each
    score += latex.count('\\frac') * 2.0
    score += latex.count('\\sqrt') * 2.0
    score += latex.count('\\int') * 2.0
    score += latex.count('\\sum') * 2.0
    score += latex.count('\\prod') * 2.0
    score += latex.count('\\lim') * 2.0
    
    # D. Grouping depth (approximated by number of opening brackets)
    # +0.5 points each
    score += latex.count('{') * 0.5 

    # 3. CLASSIFY BASED ON SCORE
    # Score < 4: Very basic math (e.g., "x^2 + y = 3") -> Simple
    # Score 4 to 12: Standard high school / college math -> Medium
    # Score > 12: Heavily nested/long nightmare math -> Complex
    if score < 4.0:
        return 'simple'
    elif score < 12.0:
        return 'medium'
    else:
        return 'complex'


def should_discard(latex: str) -> bool:
    """Check if a LaTeX string contains structures too complex for this training regime."""
    for pattern in DISCARD_PATTERNS:
        if re.search(pattern, latex):
            return True
    return False


def normalize_latex(latex: str) -> Optional[str]:
    """
    Normalize a LaTeX string for consistent training.

    Returns None if the sample should be discarded (truly unparseable).
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

    # Standardize \\over → \\frac{}{} where possible
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
    complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}

    for sample in corpus:
        latex = sample.get('latex', '')
        cleaned = normalize_latex(latex)

        if cleaned is None:
            discarded += 1
            continue

        complexity = get_complexity(cleaned)
        complexity_counts[complexity] += 1

        new_sample = dict(sample)
        new_sample['latex'] = cleaned
        new_sample['complexity'] = complexity
        normalized.append(new_sample)

    total = max(len(corpus), 1)
    logger.info(
        f"Normalized corpus: {len(normalized)} kept, {discarded} discarded "
        f"({100*discarded/total:.1f}% filtered)"
    )
    logger.info(
        f"  Complexity breakdown: "
        f"simple={complexity_counts['simple']}, "
        f"medium={complexity_counts['medium']}, "
        f"complex={complexity_counts['complex']}"
    )

    return normalized