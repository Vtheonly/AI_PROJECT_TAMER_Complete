"""
Evaluation metrics for Math OCR.

- ExpRate: Exact match rate after stripping spaces
- Edit Distance: Token-level Levenshtein distance
- Symbol Error Rate (SER): Token-level error rate
"""

import re
from typing import List, Dict


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize LaTeX string into commands and individual characters for accurate Edit Distance.
    Example: '\\frac{a}{b}' -> ['\\frac', '{', 'a', '}', '{', 'b', '}']
    """
    # Remove spaces first to standardize
    clean_latex = latex.replace(' ', '')
    # Regex: match \command or a single character
    tokens = re.findall(r'\\[a-zA-Z]+|.', clean_latex)
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