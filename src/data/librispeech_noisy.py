"""LibriSpeech test-other with additive noise.

Supports two noise modes:
  1. Gaussian noise (default) -- no extra downloads needed
  2. MS-SNSD environmental noise -- optional, richer noise types
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torchaudio
import whisper

# Use soundfile backend to avoid torchcodec/FFmpeg dependency
torchaudio.set_audio_backend("soundfile")
from torch.utils.data import Dataset

NOISE_TYPES_GAUSSIAN = [
    "gaussian_white",
    "gaussian_pink",
    "gaussian_brown",
]

NOISE_TYPES_SNSD = [
    "AirConditioner",
    "AirportAnnouncements",
    "Babble",
    "CopyMachine",
    "Munching",
    "Neighbors",
    "ShuttingDoor",
    "Typing",
]


def _generate_gaussian_noise(length: int, color: str = "white") -> torch.Tensor:
    """Generate synthetic noise of the specified color."""
    noise = torch.randn(length)
    if color == "pink":
        # Simple 1/f approximation via cumulative sum + normalization
        noise = torch.cumsum(noise, dim=0)
        noise = noise - noise.mean()
        noise = noise / (noise.std() + 1e-8)
    elif color == "brown":
        noise = torch.cumsum(torch.cumsum(noise, dim=0), dim=0)
        noise = noise - noise.mean()
        noise = noise / (noise.std() + 1e-8)
    return noise


class NoisyLibriSpeechDataset(Dataset):
    """LibriSpeech test-other with additive noise at configurable SNR.

    Directory structure expected:
        data_root/
            LibriSpeech/test-other/   (standard LibriSpeech layout)
            MS-SNSD/noise_test/       (optional, for environmental noise)
    """

    def __init__(
        self,
        data_root: str = "data",
        snr_db: float = 10.0,
        noise_type: str | None = None,
        noise_source: str = "gaussian",  # "gaussian" or "snsd"
        max_samples: int | None = None,
    ) -> None:
        self.snr_db = snr_db
        self.noise_type = noise_type
        self.noise_source = noise_source
        self.data_root = Path(data_root)
        self.librispeech_root = self.data_root / "LibriSpeech" / "test-other"

        self.samples: list[dict] = []
        self._load_transcripts()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Load SNSD noise files if available and requested
        self.snsd_noise_files: dict[str, torch.Tensor] = {}
        if noise_source == "snsd":
            noise_root = self.data_root / "MS-SNSD" / "noise_test"
            if noise_root.exists():
                for ntype in NOISE_TYPES_SNSD:
                    noise_path = noise_root / f"{ntype}.wav"
                    if noise_path.exists():
                        waveform, sr = torchaudio.load(str(noise_path))
                        if sr != 16000:
                            waveform = torchaudio.functional.resample(waveform, sr, 16000)
                        self.snsd_noise_files[ntype] = waveform[0]
            if not self.snsd_noise_files:
                print("Warning: MS-SNSD not found, falling back to Gaussian noise.")
                self.noise_source = "gaussian"

    def _load_transcripts(self) -> None:
        """Walk LibriSpeech directory to find all audio + transcript pairs."""
        if not self.librispeech_root.exists():
            print(f"Warning: {self.librispeech_root} not found. Using empty dataset.")
            return

        for speaker_dir in sorted(self.librispeech_root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                trans_file = (
                    chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                )
                if not trans_file.exists():
                    continue
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            audio_path = chapter_dir / f"{utt_id}.flac"
                            if audio_path.exists():
                                self.samples.append(
                                    {
                                        "id": utt_id,
                                        "audio_path": str(audio_path),
                                        "text": text,
                                    }
                                )

    def _add_noise(self, waveform: torch.Tensor, noise_type: str) -> torch.Tensor:
        """Add noise at specified SNR."""
        if self.noise_source == "gaussian":
            color = noise_type.replace("gaussian_", "") if "gaussian_" in noise_type else "white"
            noise = _generate_gaussian_noise(len(waveform), color=color)
        elif noise_type in self.snsd_noise_files:
            noise = self.snsd_noise_files[noise_type]
            if len(noise) < len(waveform):
                repeats = (len(waveform) // len(noise)) + 1
                noise = noise.repeat(repeats)
            noise = noise[: len(waveform)]
        else:
            noise = torch.randn_like(waveform)

        signal_power = (waveform**2).mean()
        noise_power = (noise**2).mean()
        if noise_power == 0:
            return waveform
        snr_linear = 10 ** (self.snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        return waveform + scale * noise

    @property
    def available_noise_types(self) -> list[str]:
        """Return noise types available for this dataset."""
        if self.noise_source == "snsd" and self.snsd_noise_files:
            return list(self.snsd_noise_files.keys())
        return NOISE_TYPES_GAUSSIAN

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["audio_path"])
        waveform = waveform[0]

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        noise_type = self.noise_type or random.choice(self.available_noise_types)
        waveform_noisy = self._add_noise(waveform, noise_type)

        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(waveform_noisy))

        return {
            "id": sample["id"],
            "mel": mel,
            "audio": waveform_noisy,
            "text": sample["text"],
            "noise_type": noise_type,
        }
