"""
Dataset Health Auditor.
Runs before training to guarantee the training source is not malformed.
"""

import json
import os
import logging
from collections import Counter
from typing import List

logger = logging.getLogger("TAMER.Audit")


class DatasetAuditor:
    def __init__(self, tokenizer, data_dir: str, sanitized_dir: str):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.sanitized_dir = sanitized_dir
        self.vocab = tokenizer.vocab

    def audit(
        self,
        datasets: List[str],
        min_samples: int = 1000,
        max_unk_rate: float = 0.001,
        max_missing_img_rate: float = 0.05,
    ) -> bool:
        total_samples = 0
        total_missing_img = 0
        total_empty_latex = 0
        total_unk = 0
        total_toks = 0
        complexity = Counter()
        lengths = []

        for ds in datasets:
            path = os.path.join(self.sanitized_dir, f"{ds}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"CRITICAL: {path} does not exist")

            cnt = missing = empty = long = 0
            ds_unk = 0
            ds_toks = 0

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    cnt += 1
                    latex = s.get("latex", "")
                    img_rel = s.get("image", "")

                    if not latex:
                        empty += 1
                        continue
                    if not img_rel:
                        missing += 1
                        continue

                    img_abs = (
                        os.path.join(self.data_dir, img_rel)
                        if not os.path.isabs(img_rel)
                        else img_rel
                    )
                    if not os.path.exists(img_abs):
                        missing += 1

                    toks = self.tokenizer.tokenize(latex)
                    lengths.append(len(toks))
                    ds_toks += len(toks)
                    if len(toks) > 200:
                        long += 1
                    for t in toks:
                        if t not in self.vocab:
                            ds_unk += 1

                    from .latex_normalizer import get_complexity

                    complexity[get_complexity(latex)] += 1

            total_samples += cnt
            total_missing_img += missing
            total_empty_latex += empty
            total_unk += ds_unk
            total_toks += ds_toks

            logger.info(
                f"Audit {ds:<12}: samples={cnt:>6,} missing_img={missing:>4} "
                f"empty={empty:>3} long={long:>3} unk={ds_unk}"
            )

            if cnt < min_samples:
                logger.warning(f"{ds} has only {cnt} samples (min {min_samples})")

        missing_rate = total_missing_img / max(total_samples, 1)
        unk_rate = total_unk / max(total_toks, 1)
        avg_len = sum(lengths) / max(len(lengths), 1)

        logger.info(
            f"Audit TOTAL: samples={total_samples:,} "
            f"missing_img_rate={missing_rate:.4%} "
            f"unk_rate={unk_rate:.4%} avg_len={avg_len:.1f}"
        )
        logger.info(f"Audit complexity: {dict(complexity)}")

        assert total_samples > 0, "CRITICAL: Zero samples loaded"
        assert (
            missing_rate < max_missing_img_rate
        ), f"CRITICAL: Missing image rate {missing_rate:.2%}"
        assert (
            unk_rate < max_unk_rate
        ), f"CRITICAL: UNK rate {unk_rate:.4%} — tokenizer/vocab mismatch"
        assert complexity["simple"] > 0, "CRITICAL: No simple samples for curriculum"

        logger.info("✅ Dataset audit PASSED")
        return True