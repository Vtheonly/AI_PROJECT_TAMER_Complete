"""
Evaluation metrics for Math OCR.

v2.3 Changes:
  - Added structural accuracy metrics: separate ExpRate by formula complexity
    (simple/medium/complex) so you can see exactly where the model struggles.
  - Added structural token recall: measures how well the model predicts
    row separators (\\\\), column separators (&), and environment tokens.
  - Updated tokenize_latex to properly handle \\\\, \\begin{env}, \\end{env}
    as atomic tokens for accurate edit distance computation.

Metrics:
  - ExpRate: Exact match rate after stripping spaces
  - Edit Distance: Token-level Levenshtein distance
  - Symbol Error Rate (SER): Token-level error rate
  - Leq1: Percentage of predictions within edit distance ≤ 1
"""

import re
from typing import List, Dict, Optional


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize LaTeX string into commands and individual characters for accurate metrics.

    Handles:
      - \\\\  (double backslash) as atomic row separator
      - \\begin{env} and \\end{env} as atomic tokens
      - \\command as atomic tokens
      - Individual characters

    Example: '\\frac{a}{b}' -> ['\\frac', '{', 'a', '}', '{', 'b', '}']
    Example: '\\begin{matrix} a & b \\\\ c & d \\end{matrix}'
          -> ['\\begin{matrix}', 'a', '&', 'b', '\\\\', 'c', '&', 'd', '\\end{matrix}']
    """
    # Remove spaces first to standardize
    clean = latex.strip()
    tokens = []
    i = 0
    n = len(clean)

    while i < n:
        # Skip whitespace
        if clean[i].isspace():
            i += 1
            continue

        # Double backslash (row separator) — check before single backslash
        if i + 1 < n and clean[i] == '\\' and clean[i + 1] == '\\':
            if i + 2 >= n or not clean[i + 2].isalpha():
                tokens.append('\\\\')
                i += 2
                continue

        # \begin{env} and \end{env} as atomic tokens
        if clean[i] == '\\' and (clean[i:].startswith('\\begin') or clean[i:].startswith('\\end')):
            cmd_len = 6 if clean[i:].startswith('\\begin') else 4
            j = i + cmd_len
            while j < n and clean[j].isspace():
                j += 1
            if j < n and clean[j] == '{':
                k = j + 1
                while k < n and clean[k] != '}':
                    k += 1
                if k < n:
                    token = clean[i:k + 1]
                    # Normalize internal whitespace
                    token = token[:cmd_len] + token[cmd_len:].replace(' ', '')
                    tokens.append(token)
                    i = k + 1
                    continue

        # LaTeX commands
        if clean[i] == '\\':
            j = i + 1
            while j < n and clean[j].isalpha():
                j += 1
            if j == i + 1 and j < n:
                tokens.append(clean[i:j + 1])
                i = j + 1
            else:
                tokens.append(clean[i:j])
                i = j
            continue

        # Everything else
        tokens.append(clean[i])
        i += 1

    return tokens


def edit_distance(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


def calculate_metrics(pred_latex: str, gt_latex: str) -> Dict[str, float]:
    """
    Calculate evaluation metrics between predicted and ground truth LaTeX.

    Args:
        pred_latex: Predicted LaTeX string
        gt_latex: Ground truth LaTeX string

    Returns:
        Dict with 'exact_match', 'edit_dist', 'ser', 'leq1'
    """
    # ExpRate: Exact match ignoring spaces
    pred_clean = pred_latex.replace(' ', '')
    gt_clean = gt_latex.replace(' ', '')

    exact_match = float(pred_clean == gt_clean)

    # Edit Distance: Token-level to preserve LaTeX semantics
    pred_tokens = tokenize_latex(pred_latex)
    gt_tokens = tokenize_latex(gt_latex)

    dist = edit_distance(pred_tokens, gt_tokens)
    leq1 = float(dist <= 1)

    # Symbol Error Rate (SER)
    gt_len = max(len(gt_tokens), 1)
    ser = dist / gt_len

    return {
        'exact_match': exact_match,
        'edit_dist': float(dist),
        'ser': ser,
        'leq1': leq1,
    }


def compute_batch_metrics(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """Compute average metrics over a batch of predictions."""
    if not preds or not targets:
        return {'exact_match': 0.0, 'edit_dist': 0.0, 'ser': 0.0, 'leq1': 0.0}

    all_metrics = [calculate_metrics(p, t) for p, t in zip(preds, targets)]

    n = len(all_metrics)
    return {
        'exact_match': sum(m['exact_match'] for m in all_metrics) / n,
        'edit_dist': sum(m['edit_dist'] for m in all_metrics) / n,
        'ser': sum(m['ser'] for m in all_metrics) / n,
        'leq1': sum(m['leq1'] for m in all_metrics) / n,
    }


def evaluate_structural_accuracy(
    predictions: List[str],
    targets: List[str],
) -> Dict[str, float]:
    """
    Compute structure-aware metrics, separated by formula complexity.

    Returns metrics like:
      - exprate_simple, exprate_medium, exprate_complex
      - count_simple, count_medium, count_complex
      - structural_token_recall (how well \\\\, &, \\begin, \\end are predicted)
    """
    # Lazy import to avoid circular dependency
    from ..data.latex_normalizer import get_complexity

    results = {
        'simple': {'correct': 0, 'total': 0},
        'medium': {'correct': 0, 'total': 0},
        'complex': {'correct': 0, 'total': 0},
    }

    for pred, target in zip(predictions, targets):
        complexity = get_complexity(target)
        results[complexity]['total'] += 1

        pred_clean = pred.replace(' ', '')
        tgt_clean = target.replace(' ', '')

        if pred_clean == tgt_clean:
            results[complexity]['correct'] += 1

    metrics = {}
    for level, counts in results.items():
        total = counts['total']
        if total > 0:
            metrics[f'exprate_{level}'] = counts['correct'] / total
        else:
            metrics[f'exprate_{level}'] = 0.0
        metrics[f'count_{level}'] = total

    # Structural token recall:
    # How often does the model correctly predict structural tokens
    # that appear in the ground truth?
    structural_tokens = {'\\\\', '&'}
    env_prefixes = {'\\begin', '\\end'}

    struct_correct = 0
    struct_total = 0

    for pred, target in zip(predictions, targets):
        pred_toks = tokenize_latex(pred)
        tgt_toks = tokenize_latex(target)

        for tok in tgt_toks:
            is_structural = (
                tok in structural_tokens or
                any(tok.startswith(p + '{') for p in env_prefixes)
            )
            if is_structural:
                struct_total += 1
                if tok in pred_toks:
                    struct_correct += 1

    if struct_total > 0:
        metrics['structural_token_recall'] = struct_correct / struct_total
    else:
        metrics['structural_token_recall'] = 1.0  # No structural tokens → perfect

    return metrics