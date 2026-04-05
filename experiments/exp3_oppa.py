"""Experiment 3: OPPA — Online Persistent Prompt Adaptation.

Hypothesis: Accumulating prompt knowledge across correlated samples via EMA
outperforms per-sample reset, especially for same-speaker or same-noise data.

Ablations:
  A) Single-sample (baseline, reset per sample)
  B) Persistent EMA (decay=0.9), no domain grouping
  C) Persistent EMA (decay=0.7), faster adaptation
  D) Persistent with domain grouping (separate prompt per noise/speaker)
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.whisper_wrapper import WhisperWithPrompt
from src.rewards.clap_reward import CLAPReward
from src.rl.reinforce import REINFORCE
from src.adaptation.single_sample import SingleSampleAdapter
from src.adaptation.persistent import PersistentPromptAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset, NOISE_TYPES
from src.data.l2arctic import L2ArcticDataset, L1_SPEAKERS
from src.utils import compute_wer, ExperimentLogger, load_config


def create_adapter(adapt_cfg, model, reward_fn, rl, cfg, device):
    """Create adapter from config."""
    adapt_type = adapt_cfg["type"]
    common_kwargs = dict(
        model=model,
        reward_fn=reward_fn,
        rl_optimizer=rl,
        n_candidates=cfg["decoding"]["n_candidates"],
        temp_range=tuple(cfg["decoding"]["temp_range"]),
        device=device,
    )

    if adapt_type == "single_sample":
        return SingleSampleAdapter(**common_kwargs)
    elif adapt_type == "persistent":
        return PersistentPromptAdapter(
            **common_kwargs,
            ema_decay=adapt_cfg.get("ema_decay", 0.9),
            warmup_samples=adapt_cfg.get("warmup_samples", 5),
            warmup_decay=adapt_cfg.get("warmup_decay", 0.5),
            use_domain_grouping=adapt_cfg.get("use_domain_grouping", False),
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapt_type}")


def get_domain(sample: dict) -> str | None:
    """Extract domain label from sample for grouping."""
    if "noise_type" in sample:
        return sample["noise_type"]
    if "l1_background" in sample:
        return sample["l1_background"]
    return None


def run_adaptation_config(adapt_cfg, model, reward_fn, dataset, cfg, device):
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
        experiment_name=f"{config_name}_{dataset.name}",
    )

    results = []
    domain_results = defaultdict(list)

    for i in range(len(dataset)):
        sample = dataset[i]
        domain = get_domain(sample)

        t0 = time.time()
        # Pass domain for persistent adapter
        if isinstance(adapter, PersistentPromptAdapter):
            output = adapter.adapt_and_decode(
                mel=sample["mel"], audio=sample["audio"], domain=domain
            )
        else:
            output = adapter.adapt_and_decode(
                mel=sample["mel"], audio=sample["audio"]
            )
        latency = time.time() - t0

        wer_before = compute_wer(output["baseline_text"], sample["text"])
        wer_after = compute_wer(output["text"], sample["text"])

        entry = {
            "id": sample["id"],
            "wer_before": wer_before,
            "wer": wer_after,
            "latency": latency,
            "adapter": config_name,
            "sample_idx": i,
        }
        if domain:
            entry["domain"] = domain

        # Track persistent-specific metrics
        info = output.get("info", {})
        if "ema_updated" in info:
            entry["ema_updated"] = info["ema_updated"]
            entry["reward_improvement"] = info["reward_improvement"]

        logger.log(entry)
        results.append(entry)
        if domain:
            domain_results[domain].append(entry)

    logger.save()

    # Analyze adaptation over time (do later samples benefit from persistence?)
    if len(results) > 20:
        first_half = results[: len(results) // 2]
        second_half = results[len(results) // 2 :]
        wer_first = sum(r["wer"] for r in first_half) / len(first_half)
        wer_second = sum(r["wer"] for r in second_half) / len(second_half)
    else:
        wer_first = wer_second = None

    return results, domain_results, (wer_first, wer_second)


def main():
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
    print("Experiment 3: OPPA — Online Persistent Prompt Adaptation")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )
    reward_fn = CLAPReward(device=device)

    # Prepare datasets — sorted by domain for persistence to matter
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

    all_results = {}

    for adapt_cfg in cfg["adaptation"]["configurations"]:
        config_name = adapt_cfg["name"]
        print(f"\n{'='*40}")
        print(f"Strategy: {config_name}")
        print(f"{'='*40}")

        config_results = []

        for dataset in datasets:
            print(f"\n  Dataset: {dataset.name} ({len(dataset)} samples)")
            results, domain_results, (wer_first, wer_second) = (
                run_adaptation_config(
                    adapt_cfg, model, reward_fn, dataset, cfg, device
                )
            )
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
            avg_latency = sum(r["latency"] for r in results) / len(results)

            print(
                f"  WER: {avg_wer_before:.4f} -> {avg_wer:.4f} "
                f"(delta: {avg_wer - avg_wer_before:+.4f}) "
                f"Latency: {avg_latency:.3f}s"
            )

            # Show temporal trend for persistent adapters
            if wer_first is not None and wer_second is not None:
                print(
                    f"  Temporal: 1st half WER={wer_first:.4f}, "
                    f"2nd half WER={wer_second:.4f} "
                    f"(improvement: {wer_first - wer_second:+.4f})"
                )

            config_results.extend(results)

        all_results[config_name] = config_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':25s} {'Mean WER':>10s} {'WER Delta':>10s} {'Latency':>10s}")
    print("-" * 55)
    for config_name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        delta = avg_wer - avg_wer_before
        print(
            f"  {config_name:23s} {avg_wer:10.4f} {delta:+10.4f} {avg_latency:10.3f}s"
        )


if __name__ == "__main__":
    main()
