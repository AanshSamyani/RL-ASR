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

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set GPU before torch import
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset
from src.rewards.clap_reward import CLAPReward
from src.rl.grpo import GRPO
from src.rl.reinforce import REINFORCE
from src.utils import ExperimentLogger, compute_wer, load_config
from src.whisper_wrapper import WhisperWithPrompt


def create_rl_optimizer(method_cfg: dict) -> REINFORCE | GRPO:
    """Create RL optimizer from config."""
    name = method_cfg["name"]
    if name == "reinforce":
        return REINFORCE(
            base_lr=method_cfg.get("base_lr", 1e-5),
            prompt_lr_scale=method_cfg.get("prompt_lr_scale", 100.0),
        )
    return GRPO(
        base_lr=method_cfg.get("base_lr", 1e-5),
        prompt_lr_scale=method_cfg.get("prompt_lr_scale", 100.0),
        clip_eps=method_cfg.get("clip_eps", 0.2),
        kl_coeff=method_cfg.get("kl_coeff", 0.01),
        token_level=method_cfg.get("token_level", False),
    )


def run_method_on_dataset(
    method_cfg: dict,
    model: WhisperWithPrompt,
    reward_fn: CLAPReward,
    dataset: NoisyLibriSpeechDataset,
    cfg: dict,
    device: str,
) -> list[dict]:
    """Run one RL method on one dataset."""
    method_name = method_cfg["name"]

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
        experiment_name=f"{method_name}_{dataset.noise_type or 'mixed'}",
    )

    results: list[dict] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        t0 = time.time()
        output = adapter.adapt_and_decode(mel=sample["mel"], audio=sample["audio"])
        latency = time.time() - t0

        entry = {
            "id": sample["id"],
            "wer_before": compute_wer(output["baseline_text"], sample["text"]),
            "wer": compute_wer(output["text"], sample["text"]),
            "latency": latency,
            "method": method_name,
            "noise_type": sample.get("noise_type", "unknown"),
        }
        if "kl_loss" in output.get("info", {}):
            entry["kl_loss"] = output["info"]["kl_loss"]
            entry["pg_loss"] = output["info"]["pg_loss"]

        logger.log(entry)
        results.append(entry)

    logger.save()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: GRPO vs REINFORCE")
    parser.add_argument("--config", default="configs/exp1_grpo.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="GPU id (e.g. 0, 1, 2, 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.max_samples:
        cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("Experiment 1: GRPO vs REINFORCE for ASR-TTA")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )
    reward_fn = CLAPReward(device=device)

    # Build datasets -- one per noise type for richer comparison
    noise_types = cfg["data"].get("noise_types", ["gaussian_white", "gaussian_pink", "gaussian_brown"])
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
    for method_cfg in cfg["rl"]["methods"]:
        method_name = method_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Method: {method_name}")
        print(f"{'='*40}")

        method_results: list[dict] = []
        for dataset in datasets:
            print(f"\n  Noise: {dataset.noise_type or 'mixed'} ({len(dataset)} samples)")
            results = run_method_on_dataset(
                method_cfg, model, reward_fn, dataset, cfg, device
            )
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_before = sum(r["wer_before"] for r in results) / len(results)
            avg_lat = sum(r["latency"] for r in results) / len(results)
            print(
                f"  WER: {avg_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_before:+.4f}) "
                f"Latency: {avg_lat:.3f}s"
            )
            method_results.extend(results)

        all_results[method_name] = method_results

    print("\n" + "=" * 60)
    print("SUMMARY: Mean WER across all noise types")
    print("=" * 60)
    for method_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_before = sum(r["wer_before"] for r in results) / len(results)
        avg_lat = sum(r["latency"] for r in results) / len(results)
        print(
            f"  {method_name:25s}  "
            f"WER: {avg_wer:.4f} (from {avg_before:.4f})  "
            f"Latency: {avg_lat:.3f}s"
        )


if __name__ == "__main__":
    main()
