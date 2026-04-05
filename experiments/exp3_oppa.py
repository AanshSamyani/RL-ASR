"""Experiment 3: OPPA -- Online Persistent Prompt Adaptation.

Hypothesis: Accumulating prompt knowledge across correlated samples via EMA
outperforms per-sample reset, especially within the same noise condition.

Ablations:
  A) Single-sample (baseline, reset per sample)
  B) Persistent EMA (decay=0.9), no domain grouping
  C) Persistent EMA (decay=0.7), faster adaptation
  D) Persistent with domain grouping (separate prompt per noise type)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.persistent import PersistentPromptAdapter
from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset
from src.rewards.clap_reward import CLAPReward
from src.rl.reinforce import REINFORCE
from src.utils import ExperimentLogger, compute_wer, load_config
from src.whisper_wrapper import WhisperWithPrompt


def create_adapter(
    adapt_cfg: dict,
    model: WhisperWithPrompt,
    reward_fn: CLAPReward,
    rl: REINFORCE,
    cfg: dict,
    device: str,
) -> SingleSampleAdapter | PersistentPromptAdapter:
    """Create adapter from config."""
    common = {
        "model": model,
        "reward_fn": reward_fn,
        "rl_optimizer": rl,
        "n_candidates": cfg["decoding"]["n_candidates"],
        "temp_range": tuple(cfg["decoding"]["temp_range"]),
        "device": device,
    }

    if adapt_cfg["type"] == "single_sample":
        return SingleSampleAdapter(**common)
    return PersistentPromptAdapter(
        **common,
        ema_decay=adapt_cfg.get("ema_decay", 0.9),
        warmup_samples=adapt_cfg.get("warmup_samples", 5),
        warmup_decay=adapt_cfg.get("warmup_decay", 0.5),
        use_domain_grouping=adapt_cfg.get("use_domain_grouping", False),
    )


def run_adaptation_config(
    adapt_cfg: dict,
    model: WhisperWithPrompt,
    reward_fn: CLAPReward,
    dataset: NoisyLibriSpeechDataset,
    cfg: dict,
    device: str,
) -> tuple[list[dict], dict[str, list[dict]], tuple[float | None, float | None]]:
    """Run one adaptation strategy on a dataset."""
    config_name = adapt_cfg["name"]

    rl = REINFORCE(
        base_lr=cfg["rl"].get("base_lr", 1e-5),
        prompt_lr_scale=cfg["rl"].get("prompt_lr_scale", 100.0),
    )
    rl.setup_optimizer(model.get_trainable_params())

    adapter = create_adapter(adapt_cfg, model, reward_fn, rl, cfg, device)

    logger = ExperimentLogger(
        log_dir=cfg["logging"]["results_dir"],
        experiment_name=f"{config_name}_{dataset.noise_type or 'mixed'}",
    )

    results: list[dict] = []
    domain_results: dict[str, list[dict]] = defaultdict(list)

    for i in range(len(dataset)):
        sample = dataset[i]
        domain = sample.get("noise_type")
        t0 = time.time()

        if isinstance(adapter, PersistentPromptAdapter):
            output = adapter.adapt_and_decode(
                mel=sample["mel"], audio=sample["audio"], domain=domain
            )
        else:
            output = adapter.adapt_and_decode(mel=sample["mel"], audio=sample["audio"])
        latency = time.time() - t0

        entry = {
            "id": sample["id"],
            "wer_before": compute_wer(output["baseline_text"], sample["text"]),
            "wer": compute_wer(output["text"], sample["text"]),
            "latency": latency,
            "adapter": config_name,
            "sample_idx": i,
        }
        if domain:
            entry["domain"] = domain

        info = output.get("info", {})
        if "ema_updated" in info:
            entry["ema_updated"] = info["ema_updated"]
            entry["reward_improvement"] = info["reward_improvement"]

        logger.log(entry)
        results.append(entry)
        if domain:
            domain_results[domain].append(entry)

    logger.save()

    # Temporal analysis: do later samples benefit from persistence?
    wer_first: float | None = None
    wer_second: float | None = None
    if len(results) > 20:
        first_half = results[: len(results) // 2]
        second_half = results[len(results) // 2 :]
        wer_first = sum(r["wer"] for r in first_half) / len(first_half)
        wer_second = sum(r["wer"] for r in second_half) / len(second_half)

    return results, dict(domain_results), (wer_first, wer_second)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 3: OPPA Persistent Prompt")
    parser.add_argument("--config", default="configs/exp3_oppa.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.max_samples:
        cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("Experiment 3: OPPA -- Online Persistent Prompt Adaptation")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )
    reward_fn = CLAPReward(device=device)

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

    for adapt_cfg in cfg["adaptation"]["configurations"]:
        config_name = adapt_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Strategy: {config_name}")
        print(f"{'='*40}")

        config_results: list[dict] = []
        for dataset in datasets:
            print(f"\n  Noise: {dataset.noise_type or 'mixed'} ({len(dataset)} samples)")
            results, _domain_results, (wer_first, wer_second) = run_adaptation_config(
                adapt_cfg, model, reward_fn, dataset, cfg, device
            )
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_before = sum(r["wer_before"] for r in results) / len(results)
            avg_lat = sum(r["latency"] for r in results) / len(results)

            print(
                f"  WER: {avg_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_before:+.4f}) "
                f"Latency: {avg_lat:.3f}s"
            )
            if wer_first is not None and wer_second is not None:
                print(
                    f"  Temporal: 1st half={wer_first:.4f}, "
                    f"2nd half={wer_second:.4f} "
                    f"(trend: {wer_first - wer_second:+.4f})"
                )

            config_results.extend(results)

        all_results[config_name] = config_results

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Strategy':25s} {'Mean WER':>10s} {'WER Delta':>10s} {'Latency':>10s}"
    print(header)
    print("-" * len(header))
    for config_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_before = sum(r["wer_before"] for r in results) / len(results)
        avg_lat = sum(r["latency"] for r in results) / len(results)
        delta = avg_wer - avg_before
        print(f"  {config_name:23s} {avg_wer:10.4f} {delta:+10.4f} {avg_lat:10.3f}s")


if __name__ == "__main__":
    main()
