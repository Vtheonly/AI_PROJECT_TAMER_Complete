from typing import List

def edit_distance(s1: List[str], s2: List[str]) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def calculate_metrics(pred_latex: str, gt_latex: str) -> dict:
    pred_clean = pred_latex.replace(' ', '')
    gt_clean = gt_latex.replace(' ', '')
    
    exact_match = int(pred_clean == gt_clean)
    
    dist = edit_distance(list(pred_clean), list(gt_clean))
    leq1 = int(dist <= 1)
    
    bracket_correct = int(pred_clean.count('{') == pred_clean.count('}'))
    
    return {
        'exact': exact_match,
        'leq1': leq1,
        'bracket': bracket_correct
    }