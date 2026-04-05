"""LibriSpeech test-other with additive noise from MS-SNSD."""

import os
import random
from pathlib import Path

import torch
import torchaudio
import whisper
from torch.utils.data import Dataset


NOISE_TYPES = [
    "AirConditioner",
    "AirportAnnouncements",
    "Babble",
    "CopyMachine",
    "Munching",
    "Neighbors",
    "ShuttingDoor",
    "Typing",
]


class NoisyLibriSpeechDataset(Dataset):
    """LibriSpeech test-other with additive noise at configurable SNR.

    Directory structure expected:
        data_root/
            LibriSpeech/test-other/          (standard LibriSpeech layout)
            MS-SNSD/noise_test/              (noise files from MS-SNSD)
    """

    def __init__(
        self,
        data_root: str = "data",
        snr_db: float = 10.0,
        noise_type: str | None = None,
        max_samples: int | None = None,
    ):
        self.snr_db = snr_db
        self.noise_type = noise_type
        self.data_root = Path(data_root)
        self.librispeech_root = self.data_root / "LibriSpeech" / "test-other"
        self.noise_root = self.data_root / "MS-SNSD" / "noise_test"

        # Collect all utterances
        self.samples = []
        self._load_transcripts()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Load noise files
        self.noise_files = {}
        if self.noise_root.exists():
            for ntype in NOISE_TYPES:
                noise_path = self.noise_root / f"{ntype}.wav"
                if noise_path.exists():
                    waveform, sr = torchaudio.load(str(noise_path))
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    self.noise_files[ntype] = waveform[0]  # mono

    def _load_transcripts(self):
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
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            audio_path = chapter_dir / f"{utt_id}.flac"
                            if audio_path.exists():
                                self.samples.append({
                                    "id": utt_id,
                                    "audio_path": str(audio_path),
                                    "text": text,
                                })

    def _add_noise(self, waveform: torch.Tensor, noise_type: str) -> torch.Tensor:
        """Add noise at specified SNR."""
        if noise_type not in self.noise_files:
            return waveform

        noise = self.noise_files[noise_type]
        # Tile or trim noise to match waveform length
        if len(noise) < len(waveform):
            repeats = (len(waveform) // len(noise)) + 1
            noise = noise.repeat(repeats)
        noise = noise[: len(waveform)]

        # Compute scaling for target SNR
        signal_power = (waveform**2).mean()
        noise_power = (noise**2).mean()
        if noise_power == 0:
            return waveform
        snr_linear = 10 ** (self.snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))

        return waveform + scale * noise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["audio_path"])
        waveform = waveform[0]  # mono

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Add noise
        noise_type = self.noise_type or random.choice(NOISE_TYPES)
        if self.noise_files:
            waveform = self._add_noise(waveform, noise_type)

        # Convert to mel spectrogram
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(waveform)
        )

        return {
            "id": sample["id"],
            "mel": mel,
            "audio": waveform,
            "text": sample["text"],
            "noise_type": noise_type,
        }
