"""Baseline reproduction: ASR-TRA with REINFORCE + CLAP + single-sample."""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.whisper_wrapper import WhisperWithPrompt
from src.rewards.clap_reward import CLAPReward
from src.rl.reinforce import REINFORCE
from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset, NOISE_TYPES
from src.data.l2arctic import L2ArcticDataset, L1_SPEAKERS
from src.utils import compute_wer, ExperimentLogger, load_config


def run_on_dataset(adapter, dataset, logger, verbose=True):
    """Run adaptation on a full dataset."""
    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        t0 = time.time()
        output = adapter.adapt_and_decode(
            mel=sample["mel"],
            audio=sample["audio"],
        )
        latency = time.time() - t0

        wer_before = compute_wer(output["baseline_text"], sample["text"])
        wer_after = compute_wer(output["text"], sample["text"])

        entry = {
            "id": sample["id"],
            "reference": sample["text"],
            "baseline": output["baseline_text"],
            "adapted": output["text"],
            "wer_before": wer_before,
            "wer": wer_after,
            "latency": latency,
            "mean_reward": output["info"].get("mean_reward", 0),
        }

        # Add domain info if available
        if "noise_type" in sample:
            entry["noise_type"] = sample["noise_type"]
        if "l1_background" in sample:
            entry["l1_background"] = sample["l1_background"]

        logger.log(entry)
        results.append(entry)

        if verbose and (i + 1) % 10 == 0:
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_wer_before = sum(r["wer_before"] for r in results) / len(results)
            print(
                f"  [{i+1}/{len(dataset)}] "
                f"WER: {avg_wer_before:.3f} -> {avg_wer:.3f} "
                f"({latency:.2f}s/utt)"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="ASR-TRA Baseline Reproduction")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", choices=["librispeech", "l2arctic", "both"], default="both")
    parser.add_argument("--noise-type", default=None, help="Specific noise type")
    parser.add_argument("--l1", default=None, help="Specific L1 background")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ASR-TRA Baseline Reproduction")
    print("=" * 60)

    # Initialize model
    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )

    # Initialize reward
    reward_fn = CLAPReward(device=device)

    # Initialize RL
    rl = REINFORCE(base_lr=1e-5, prompt_lr_scale=100.0)
    rl.setup_optimizer(model.get_trainable_params())

    # Initialize adapter
    adapter = SingleSampleAdapter(
        model=model,
        reward_fn=reward_fn,
        rl_optimizer=rl,
        n_candidates=cfg["decoding"]["n_candidates"],
        temp_range=tuple(cfg["decoding"]["temp_range"]),
        device=device,
    )

    max_samples = args.max_samples or cfg["data"].get("max_samples")

    # Run on LibriSpeech
    if args.dataset in ("librispeech", "both"):
        noise_types = [args.noise_type] if args.noise_type else NOISE_TYPES
        for noise in noise_types:
            print(f"\n--- LibriSpeech + {noise} (SNR={cfg['data']['snr_db']}dB) ---")
            dataset = NoisyLibriSpeechDataset(
                data_root=cfg["data"]["root"],
                snr_db=cfg["data"]["snr_db"],
                noise_type=noise,
                max_samples=max_samples,
            )
            if len(dataset) == 0:
                print(f"  No data found, skipping.")
                continue

            logger = ExperimentLogger(
                log_dir="results/baseline",
                experiment_name=f"baseline_librispeech_{noise}",
            )
            results = run_on_dataset(adapter, dataset, logger, verbose=True)
            logger.save()
            s = logger.summary()
            print(f"  Final: WER={s['mean_wer']:.4f}, Latency={s['mean_latency']:.3f}s")

    # Run on L2-Arctic
    if args.dataset in ("l2arctic", "both"):
        l1_backgrounds = [args.l1] if args.l1 else list(L1_SPEAKERS.keys())
        for l1 in l1_backgrounds:
            print(f"\n--- L2-Arctic: {l1} ---")
            dataset = L2ArcticDataset(
                data_root=cfg["data"]["root"],
                l1_background=l1,
                max_samples=max_samples,
            )
            if len(dataset) == 0:
                print(f"  No data found, skipping.")
                continue

            logger = ExperimentLogger(
                log_dir="results/baseline",
                experiment_name=f"baseline_l2arctic_{l1}",
            )
            results = run_on_dataset(adapter, dataset, logger, verbose=True)
            logger.save()
            s = logger.summary()
            print(f"  Final: WER={s['mean_wer']:.4f}, Latency={s['mean_latency']:.3f}s")


if __name__ == "__main__":
    main()
