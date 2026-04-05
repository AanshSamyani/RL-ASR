"""Baseline reproduction: ASR-TRA with REINFORCE + CLAP + single-sample."""

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
from src.rl.reinforce import REINFORCE
from src.utils import ExperimentLogger, compute_wer, load_config
from src.whisper_wrapper import WhisperWithPrompt


def run_on_dataset(
    adapter: SingleSampleAdapter,
    dataset: NoisyLibriSpeechDataset,
    logger: ExperimentLogger,
    verbose: bool = True,
) -> list[dict]:
    """Run adaptation on a full dataset."""
    results: list[dict] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        t0 = time.time()
        output = adapter.adapt_and_decode(mel=sample["mel"], audio=sample["audio"])
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
            "noise_type": sample.get("noise_type", "unknown"),
        }
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR-TRA Baseline Reproduction")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--noise-type", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id (e.g. 0, 1, 2, 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ASR-TRA Baseline Reproduction")
    print("=" * 60)

    model = WhisperWithPrompt(
        model_name=cfg["model"]["name"],
        prompt_length=cfg["model"]["prompt_length"],
        device=device,
    )
    reward_fn = CLAPReward(device=device)

    rl = REINFORCE(base_lr=1e-5, prompt_lr_scale=100.0)
    rl.setup_optimizer(model.get_trainable_params())

    adapter = SingleSampleAdapter(
        model=model,
        reward_fn=reward_fn,
        rl_optimizer=rl,
        n_candidates=cfg["decoding"]["n_candidates"],
        temp_range=tuple(cfg["decoding"]["temp_range"]),
        device=device,
    )

    max_samples = args.max_samples or cfg["data"].get("max_samples")

    dataset = NoisyLibriSpeechDataset(
        data_root=cfg["data"]["root"],
        snr_db=cfg["data"]["snr_db"],
        noise_type=args.noise_type,
        max_samples=max_samples,
    )
    if len(dataset) == 0:
        print("No data found. Run: bash scripts/setup_data.sh")
        return

    noise_label = args.noise_type or "mixed"
    print(f"\n--- LibriSpeech test-other + {noise_label} (SNR={cfg['data']['snr_db']}dB) ---")

    logger = ExperimentLogger(
        log_dir="results/baseline",
        experiment_name=f"baseline_{noise_label}",
    )
    run_on_dataset(adapter, dataset, logger, verbose=True)
    logger.save()
    s = logger.summary()
    print(f"  Final: WER={s['mean_wer']:.4f}, Latency={s['mean_latency']:.3f}s")


if __name__ == "__main__":
    main()
