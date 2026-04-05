"""Experiment 2: PARE -- Phoneme-Aware Reward Ensemble.

Hypothesis: Combining CLAP + GPT-2 perplexity + self-consistency yields
a reward signal with higher correlation to true WER than CLAP alone,
achieving LLM-level reward quality at CLAP-level latency.

Ablations:
  A) CLAP only (baseline)
  B) CLAP + LM perplexity
  C) CLAP + self-consistency
  D) Full PARE (CLAP + LM + consistency)
  E) LM-heavy PARE variant
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set GPU before torch import
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset
from src.rewards.clap_reward import CLAPReward
from src.rewards.ensemble import RewardEnsemble
from src.rl.reinforce import REINFORCE
from src.utils import ExperimentLogger, compute_wer, load_config
from src.whisper_wrapper import WhisperWithPrompt


def create_reward(
    reward_cfg: dict, device: str
) -> CLAPReward | RewardEnsemble:
    """Create reward function from config."""
    if reward_cfg.get("name") == "clap_only":
        return CLAPReward(device=device)

    return RewardEnsemble(
        use_clap=reward_cfg.get("use_clap", True),
        use_lm=reward_cfg.get("use_lm", False),
        use_consistency=reward_cfg.get("use_consistency", False),
        clap_weight=reward_cfg.get("clap_weight", 0.5),
        lm_weight=reward_cfg.get("lm_weight", 0.3),
        consistency_weight=reward_cfg.get("consistency_weight", 0.2),
        device=device,
    )


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Manual Spearman correlation (avoids hard scipy dependency)."""

    def _rankdata(arr: np.ndarray) -> np.ndarray:
        temp = np.argsort(arr)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    n = len(x)
    if n < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    d_sq = np.sum((rx - ry) ** 2)
    return 1 - (6 * d_sq) / (n * (n**2 - 1))


def run_reward_config(
    reward_cfg: dict,
    model: WhisperWithPrompt,
    dataset: NoisyLibriSpeechDataset,
    cfg: dict,
    device: str,
) -> tuple[list[dict], float]:
    """Run one reward config on one dataset, return results + correlation."""
    config_name = reward_cfg["name"]
    reward_fn = create_reward(reward_cfg, device)

    rl = REINFORCE(
        base_lr=cfg["rl"].get("base_lr", 1e-5),
        prompt_lr_scale=cfg["rl"].get("prompt_lr_scale", 100.0),
    )
    rl.setup_optimizer(model.get_trainable_params())

    adapter = SingleSampleAdapter(
        model=model,
        reward_fn=reward_fn,
        rl_optimizer=rl,
        n_candidates=cfg["decoding"]["n_candidates"],
        temp_range=tuple(cfg["decoding"]["temp_range"]),
        device=device,
    )

    logger = ExperimentLogger(
        log_dir=cfg["logging"]["results_dir"],
        experiment_name=f"{config_name}_{dataset.noise_type or 'mixed'}",
    )

    results: list[dict] = []
    reward_wer_pairs: list[tuple[float, float]] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        t0 = time.time()
        output = adapter.adapt_and_decode(mel=sample["mel"], audio=sample["audio"])
        latency = time.time() - t0

        wer_after = compute_wer(output["text"], sample["text"])
        entry = {
            "id": sample["id"],
            "wer_before": compute_wer(output["baseline_text"], sample["text"]),
            "wer": wer_after,
            "latency": latency,
            "reward_config": config_name,
            "mean_reward": output["info"].get("mean_reward", 0),
        }
        logger.log(entry)
        results.append(entry)
        reward_wer_pairs.append((output["info"].get("mean_reward", 0), wer_after))

    logger.save()

    rewards_arr = np.array([p[0] for p in reward_wer_pairs])
    wers_arr = np.array([p[1] for p in reward_wer_pairs])
    corr = _spearman_corr(rewards_arr, wers_arr)

    return results, corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2: PARE Reward Ensemble")
    parser.add_argument("--config", default="configs/exp2_pare.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="GPU id (e.g. 0, 1, 2, 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.max_samples:
        cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("Experiment 2: PARE -- Reward Ensemble Ablation")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )

    noise_types = cfg["data"].get(
        "noise_types", ["gaussian_white", "gaussian_pink", "gaussian_brown"]
    )
    datasets: list[NoisyLibriSpeechDataset] = []
    for noise in noise_types:
        ds = NoisyLibriSpeechDataset(
            data_root=cfg["data"]["root"],
            snr_db=cfg["data"]["snr_db"],
            noise_type=noise,
            max_samples=cfg["data"]["max_samples"],
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        print("No datasets found. Run: bash scripts/setup_data.sh")
        return

    all_results: dict[str, list[dict]] = {}
    all_correlations: dict[str, float] = {}

    for reward_cfg in cfg["reward"]["configurations"]:
        config_name = reward_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Reward: {config_name}")
        print(f"{'='*40}")

        config_results: list[dict] = []
        config_corrs: list[float] = []

        for dataset in datasets:
            print(f"\n  Noise: {dataset.noise_type or 'mixed'} ({len(dataset)} samples)")
            results, corr = run_reward_config(reward_cfg, model, dataset, cfg, device)
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_before = sum(r["wer_before"] for r in results) / len(results)
            avg_lat = sum(r["latency"] for r in results) / len(results)
            print(
                f"  WER: {avg_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_before:+.4f}) "
                f"Latency: {avg_lat:.3f}s  rho: {corr:.3f}"
            )
            config_results.extend(results)
            config_corrs.append(corr)

        all_results[config_name] = config_results
        all_correlations[config_name] = float(np.mean(config_corrs))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Config':25s} {'Mean WER':>10s} {'Latency':>10s} {'rho(R,WER)':>12s}"
    print(header)
    print("-" * len(header))
    for config_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_lat = sum(r["latency"] for r in results) / len(results)
        corr = all_correlations[config_name]
        print(f"  {config_name:23s} {avg_wer:10.4f} {avg_lat:10.3f}s {corr:11.3f}")


if __name__ == "__main__":
    main()
