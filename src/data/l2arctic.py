"""L2-Arctic accented English speech dataset."""

import os
from pathlib import Path

import torch
import torchaudio
import whisper
from torch.utils.data import Dataset


L1_SPEAKERS = {
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "mandarin": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "vietnamese": ["PNV", "THV", "TLV", "HNV"],
}


class L2ArcticDataset(Dataset):
    """L2-Arctic dataset for accented English evaluation.

    Directory structure expected:
        data_root/
            l2arctic_release_v5/
                <SPEAKER_ID>/
                    annotation/
                        <SPEAKER_ID>_arctic_*.txt
                    wav/
                        <SPEAKER_ID>_arctic_*.wav
    """

    def __init__(
        self,
        data_root: str = "data",
        l1_background: str | None = None,
        max_samples: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.l2arctic_root = self.data_root / "l2arctic_release_v5"
        self.l1_background = l1_background

        self.samples = []
        self._load_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self):
        """Load audio paths and transcripts."""
        if not self.l2arctic_root.exists():
            print(f"Warning: {self.l2arctic_root} not found. Using empty dataset.")
            return

        # Determine which speakers to include
        if self.l1_background:
            speakers = L1_SPEAKERS.get(self.l1_background.lower(), [])
        else:
            speakers = [s for group in L1_SPEAKERS.values() for s in group]

        for speaker_id in speakers:
            speaker_dir = self.l2arctic_root / speaker_id
            if not speaker_dir.exists():
                continue

            annotation_dir = speaker_dir / "annotation"
            wav_dir = speaker_dir / "wav"

            if not annotation_dir.exists() or not wav_dir.exists():
                continue

            for txt_file in sorted(annotation_dir.glob(f"{speaker_id}_arctic_*.txt")):
                utt_id = txt_file.stem
                wav_file = wav_dir / f"{utt_id}.wav"
                if not wav_file.exists():
                    continue

                with open(txt_file) as f:
                    text = f.read().strip()

                # Find L1 background for this speaker
                l1 = "unknown"
                for bg, spks in L1_SPEAKERS.items():
                    if speaker_id in spks:
                        l1 = bg
                        break

                self.samples.append({
                    "id": utt_id,
                    "audio_path": str(wav_file),
                    "text": text,
                    "speaker": speaker_id,
                    "l1_background": l1,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["audio_path"])
        waveform = waveform[0]  # mono

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(waveform)
        )

        return {
            "id": sample["id"],
            "mel": mel,
            "audio": waveform,
            "text": sample["text"],
            "speaker": sample["speaker"],
            "l1_background": sample["l1_background"],
        }
