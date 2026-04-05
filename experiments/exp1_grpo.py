"""Experiment 1: GRPO vs REINFORCE for test-time ASR adaptation.

Hypothesis: GRPO's normalized advantages + KL regularization + gradient
clipping yield more stable and effective test-time adaptation than
vanilla REINFORCE.

Ablations:
  A) REINFORCE (baseline reproduction)
  B) GRPO sequence-level
  C) GRPO token-level
  D) GRPO token-level with high KL penalty
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.whisper_wrapper import WhisperWithPrompt
from src.rewards.clap_reward import CLAPReward
from src.rl.reinforce import REINFORCE
from src.rl.grpo import GRPO
from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset, NOISE_TYPES
from src.data.l2arctic import L2ArcticDataset, L1_SPEAKERS
from src.utils import compute_wer, ExperimentLogger, load_config


def create_rl_optimizer(method_cfg: dict):
    """Create RL optimizer from config."""
    name = method_cfg["name"]
    if name == "reinforce":
        rl = REINFORCE(
            base_lr=method_cfg.get("base_lr", 1e-5),
            prompt_lr_scale=method_cfg.get("prompt_lr_scale", 100.0),
        )
    else:  # grpo variants
        rl = GRPO(
            base_lr=method_cfg.get("base_lr", 1e-5),
            prompt_lr_scale=method_cfg.get("prompt_lr_scale", 100.0),
            clip_eps=method_cfg.get("clip_eps", 0.2),
            kl_coeff=method_cfg.get("kl_coeff", 0.01),
            token_level=method_cfg.get("token_level", False),
        )
    return rl


def run_method_on_dataset(method_cfg, model, reward_fn, dataset, cfg, device):
    """Run one RL method on one dataset, return results."""
    method_name = method_cfg["name"]

    # Create fresh RL optimizer (resets optimizer state)
    rl = create_rl_optimizer(method_cfg)
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
        experiment_name=f"{method_name}_{dataset.name if hasattr(dataset, 'name') else 'data'}",
    )

    results = []
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
            "method": method_name,
        }
        if "noise_type" in sample:
            entry["noise_type"] = sample["noise_type"]
        if "l1_background" in sample:
            entry["l1_background"] = sample["l1_background"]

        # Add GRPO-specific diagnostics
        if "kl_loss" in output.get("info", {}):
            entry["kl_loss"] = output["info"]["kl_loss"]
            entry["pg_loss"] = output["info"]["pg_loss"]

        logger.log(entry)
        results.append(entry)

    logger.save()
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 1: GRPO vs REINFORCE")
    parser.add_argument("--config", default="configs/exp1_grpo.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.max_samples:
        cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("Experiment 1: GRPO vs REINFORCE for ASR-TTA")
    print("=" * 60)

    # Initialize model and reward (shared across methods)
    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )
    reward_fn = CLAPReward(device=device)

    # Prepare datasets
    datasets = []
    for ds_name in cfg["data"]["datasets"]:
        if ds_name == "librispeech_noisy":
            for noise in NOISE_TYPES[:3]:  # Quick: first 3 noise types
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
            for l1 in ["arabic", "mandarin", "vietnamese"]:  # Quick: 3 accents
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

    # Run each method on each dataset
    all_results = {}
    for method_cfg in cfg["rl"]["methods"]:
        method_name = method_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Method: {method_name}")
        print(f"{'='*40}")

        method_results = []
        for dataset in datasets:
            print(f"\n  Dataset: {dataset.name} ({len(dataset)} samples)")
            results = run_method_on_dataset(
                method_cfg, model, reward_fn, dataset, cfg, device
            )
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
            avg_latency = sum(r["latency"] for r in results) / len(results)
            print(
                f"  WER: {avg_wer_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_wer_before:+.4f}) "
                f"Latency: {avg_latency:.3f}s"
            )
            method_results.extend(results)

        all_results[method_name] = method_results

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Mean WER across all datasets")
    print("=" * 60)
    for method_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        print(
            f"  {method_name:25s}  "
            f"WER: {avg_wer:.4f} (from {avg_wer_before:.4f})  "
            f"Latency: {avg_latency:.3f}s"
        )


if __name__ == "__main__":
    main()
