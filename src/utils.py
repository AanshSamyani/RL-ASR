"""Utilities: WER computation, logging, config loading."""

import json
import os
import time
from dataclasses import dataclass, field

import yaml


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate between hypothesis and reference."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute character-level edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@dataclass
class ExperimentLogger:
    """Simple experiment logger that writes results to JSON."""

    log_dir: str = "results"
    experiment_name: str = "experiment"
    entries: list = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, entry: dict):
        entry["timestamp"] = time.time()
        self.entries.append(entry)

    def save(self):
        path = os.path.join(self.log_dir, f"{self.experiment_name}.json")
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)
        print(f"Results saved to {path}")

    def summary(self) -> dict:
        """Compute summary statistics from logged entries."""
        if not self.entries:
            return {}
        wers = [e["wer"] for e in self.entries if "wer" in e]
        latencies = [e["latency"] for e in self.entries if "latency" in e]
        return {
            "mean_wer": sum(wers) / len(wers) if wers else None,
            "mean_latency": sum(latencies) / len(latencies) if latencies else None,
            "n_samples": len(self.entries),
        }


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)
