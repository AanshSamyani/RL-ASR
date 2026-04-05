"""Combined experiment: Best of all three directions.

Runs the best configuration from each experiment together:
- GRPO token-level (from Exp 1)
- PARE full reward ensemble (from Exp 2)
- Persistent prompt adaptation (from Exp 3)

Tests whether improvements are orthogonal and compound.
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

from src.adaptation.persistent import PersistentPromptAdapter
from src.adaptation.single_sample import SingleSampleAdapter
from src.data.librispeech_noisy import NoisyLibriSpeechDataset
from src.rewards.clap_reward import CLAPReward
from src.rewards.ensemble import RewardEnsemble
from src.rl.grpo import GRPO
from src.rl.reinforce import REINFORCE
from src.utils import ExperimentLogger, compute_wer
from src.whisper_wrapper import WhisperWithPrompt

CONFIGURATIONS: dict[str, dict[str, str]] = {
    "baseline_asrtra": {"rl": "reinforce", "reward": "clap", "adapter": "single_sample"},
    "grpo_only": {"rl": "grpo_token", "reward": "clap", "adapter": "single_sample"},
    "pare_only": {"rl": "reinforce", "reward": "pare", "adapter": "single_sample"},
    "oppa_only": {"rl": "reinforce", "reward": "clap", "adapter": "persistent"},
    "grpo_pare": {"rl": "grpo_token", "reward": "pare", "adapter": "single_sample"},
    "grpo_oppa": {"rl": "grpo_token", "reward": "clap", "adapter": "persistent"},
    "all_combined": {"rl": "grpo_token", "reward": "pare", "adapter": "persistent"},
}


def build_components(
    config: dict[str, str],
    model: WhisperWithPrompt,
    device: str,
) -> SingleSampleAdapter | PersistentPromptAdapter:
    """Build RL, reward, and adapter from a config dict."""
    if config["rl"] == "reinforce":
        rl: REINFORCE | GRPO = REINFORCE(base_lr=1e-5, prompt_lr_scale=100.0)
    else:
        rl = GRPO(
            base_lr=1e-5,
            prompt_lr_scale=100.0,
            clip_eps=0.2,
            kl_coeff=0.01,
            token_level=True,
        )
    rl.setup_optimizer(model.get_trainable_params())

    reward_fn: CLAPReward | RewardEnsemble
    if config["reward"] == "clap":
        reward_fn = CLAPReward(device=device)
    else:
        reward_fn = RewardEnsemble(
            use_clap=True,
            use_lm=True,
            use_consistency=True,
            clap_weight=0.5,
            lm_weight=0.3,
            consistency_weight=0.2,
            device=device,
        )

    common = {
        "model": model,
        "reward_fn": reward_fn,
        "rl_optimizer": rl,
        "n_candidates": 4,
        "temp_range": (0.4, 0.6),
        "device": device,
    }
    if config["adapter"] == "single_sample":
        return SingleSampleAdapter(**common)
    return PersistentPromptAdapter(
        **common,
        ema_decay=0.9,
        warmup_samples=5,
        warmup_decay=0.5,
        use_domain_grouping=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined experiment")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base"])
    parser.add_argument("--gpu", type=str, default=None, help="GPU id (e.g. 0, 1, 2, 3)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Combined Experiment: All Improvements Together")
    print("=" * 70)

    model = WhisperWithPrompt(model_name=args.model, prompt_length=4, device=device)

    datasets: list[NoisyLibriSpeechDataset] = []
    for noise in ["gaussian_white", "gaussian_pink"]:
        ds = NoisyLibriSpeechDataset(
            data_root=args.data_root,
            snr_db=10.0,
            noise_type=noise,
            max_samples=args.max_samples,
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        print("No datasets found.")
        return

    all_results: dict[str, list[dict]] = {}
    for config_name, config in CONFIGURATIONS.items():
        print(f"\n--- {config_name} ---")
        adapter = build_components(config, model, device)

        logger = ExperimentLogger(
            log_dir="results/combined",
            experiment_name=config_name,
        )

        results: list[dict] = []
        for dataset in datasets:
            for i in range(len(dataset)):
                sample = dataset[i]
                t0 = time.time()
                if isinstance(adapter, PersistentPromptAdapter):
                    domain = sample.get("noise_type")
                    output = adapter.adapt_and_decode(
                        sample["mel"], sample["audio"], domain=domain
                    )
                else:
                    output = adapter.adapt_and_decode(sample["mel"], sample["audio"])
                latency = time.time() - t0

                entry = {
                    "wer": compute_wer(output["text"], sample["text"]),
                    "wer_before": compute_wer(output["baseline_text"], sample["text"]),
                    "latency": latency,
                }
                logger.log(entry)
                results.append(entry)

        logger.save()
        all_results[config_name] = results
        avg_wer = sum(r["wer"] for r in results) / len(results)
        print(f"  Mean WER: {avg_wer:.4f} ({len(results)} samples)")

    print("\n" + "=" * 70)
    header = (
        f"{'Configuration':20s} {'WER Before':>12s} "
        f"{'WER After':>12s} {'Delta':>10s} {'Latency':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        wb = sum(r["wer_before"] for r in results) / len(results)
        wa = sum(r["wer"] for r in results) / len(results)
        lat = sum(r["latency"] for r in results) / len(results)
        print(f"  {name:18s} {wb:12.4f} {wa:12.4f} {wa-wb:+10.4f} {lat:10.3f}s")


if __name__ == "__main__":
    main()
