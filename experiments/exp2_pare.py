"""Experiment 2: PARE — Phoneme-Aware Reward Ensemble.

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

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.whisper_wrapper import WhisperWithPrompt
from src.rewards.clap_reward import CLAPReward
from src.rewards.ensemble import RewardEnsemble
from src.rl.reinforce import REINFORCE
from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset, NOISE_TYPES
from src.data.l2arctic import L2ArcticDataset, L1_SPEAKERS
from src.utils import compute_wer, ExperimentLogger, load_config


def create_reward(reward_cfg: dict, device: str):
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


def run_reward_config(reward_cfg, model, dataset, cfg, device):
    """Run one reward configuration on one dataset."""
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
        experiment_name=f"{config_name}_{dataset.name}",
    )

    results = []
    reward_wer_pairs = []  # For correlation analysis

    for i in range(len(dataset)):
        sample = dataset[i]
        t0 = time.time()
        output = adapter.adapt_and_decode(mel=sample["mel"], audio=sample["audio"])
        latency = time.time() - t0

        wer_before = compute_wer(output["baseline_text"], sample["text"])
        wer_after = compute_wer(output["text"], sample["text"])

        entry = {
            "id": sample["id"],
            "wer_before": wer_before,
            "wer": wer_after,
            "latency": latency,
            "reward_config": config_name,
            "mean_reward": output["info"].get("mean_reward", 0),
        }

        logger.log(entry)
        results.append(entry)

        # Track reward-WER correlation
        reward_wer_pairs.append(
            (output["info"].get("mean_reward", 0), wer_after)
        )

    logger.save()

    # Compute Spearman correlation between reward and WER
    if reward_wer_pairs:
        rewards_arr = np.array([p[0] for p in reward_wer_pairs])
        wers_arr = np.array([p[1] for p in reward_wer_pairs])
        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(rewards_arr, wers_arr)
        except ImportError:
            # Manual Spearman
            def _rankdata(x):
                temp = sorted(enumerate(x), key=lambda t: t[1])
                ranks = [0.0] * len(x)
                for rank, (idx, _) in enumerate(temp):
                    ranks[idx] = rank + 1
                return ranks
            r_rewards = _rankdata(rewards_arr.tolist())
            r_wers = _rankdata(wers_arr.tolist())
            n = len(r_rewards)
            d_sq = sum((a - b) ** 2 for a, b in zip(r_rewards, r_wers))
            corr = 1 - (6 * d_sq) / (n * (n**2 - 1)) if n > 1 else 0
            pval = None
    else:
        corr, pval = 0, None

    return results, corr


def main():
    parser = argparse.ArgumentParser(description="Exp 2: PARE Reward Ensemble")
    parser.add_argument("--config", default="configs/exp2_pare.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.max_samples:
        cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("Experiment 2: PARE — Reward Ensemble Ablation")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )

    # Prepare datasets
    datasets = []
    for ds_name in cfg["data"]["datasets"]:
        if ds_name == "librispeech_noisy":
            for noise in NOISE_TYPES[:3]:
                ds = NoisyLibriSpeechDataset(
                    data_root=cfg["data"]["root"],
                    snr_db=cfg["data"]["snr_db"],
                    noise_type=noise,
                    max_samples=cfg["data"]["max_samples"],
                )
                ds.name = f"librispeech_{noise}"
                if len(ds) > 0:
                    datasets.append(ds)
        elif ds_name == "l2arctic":
            for l1 in ["arabic", "mandarin", "vietnamese"]:
                ds = L2ArcticDataset(
                    data_root=cfg["data"]["root"],
                    l1_background=l1,
                    max_samples=cfg["data"]["max_samples"],
                )
                ds.name = f"l2arctic_{l1}"
                if len(ds) > 0:
                    datasets.append(ds)

    if not datasets:
        print("No datasets found. Please run scripts/setup_data.sh first.")
        return

    # Run each reward configuration
    all_results = {}
    all_correlations = {}

    for reward_cfg in cfg["reward"]["configurations"]:
        config_name = reward_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Reward: {config_name}")
        print(f"{'='*40}")

        config_results = []
        config_corrs = []

        for dataset in datasets:
            print(f"\n  Dataset: {dataset.name} ({len(dataset)} samples)")
            results, corr = run_reward_config(
                reward_cfg, model, dataset, cfg, device
            )
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
            avg_latency = sum(r["latency"] for r in results) / len(results)
            print(
                f"  WER: {avg_wer_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_wer_before:+.4f}) "
                f"Latency: {avg_latency:.3f}s "
                f"Reward-WER corr: {corr:.3f}"
            )
            config_results.extend(results)
            config_corrs.append(corr)

        all_results[config_name] = config_results
        all_correlations[config_name] = np.mean(config_corrs)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':25s} {'Mean WER':>10s} {'Latency':>10s} {'Reward-WER rho':>15s}")
    print("-" * 60)
    for config_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        corr = all_correlations[config_name]
        print(f"  {config_name:23s} {avg_wer:10.4f} {avg_latency:10.3f}s {corr:14.3f}")


if __name__ == "__main__":
    main()
