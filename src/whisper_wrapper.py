"""Whisper model wrapper with prompt injection and stochastic decoding."""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
import whisper
from whisper.decoding import DecodingOptions, DecodingResult

from .prompt import LearnablePrompt


class WhisperWithPrompt(torch.nn.Module):
    """Wraps a Whisper model to support learnable decoder prompts
    and temperature-controlled stochastic decoding for TTA."""

    def __init__(
        self,
        model_name: str = "tiny",
        prompt_length: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        embed_dim = self.model.dims.n_text_state  # 384 for tiny, 512 for base
        self.prompt = LearnablePrompt(length=prompt_length, embed_dim=embed_dim)
        self.prompt = self.prompt.to(device)

        # Store original decoder weights for restoration
        self._original_decoder_state = None

    def save_state(self):
        """Save model + prompt state for per-sample restoration."""
        self._original_decoder_state = copy.deepcopy(
            self.model.decoder.state_dict()
        )
        self._original_prompt_state = self.prompt.clone_state()

    def restore_state(self):
        """Restore model + prompt to saved state."""
        if self._original_decoder_state is not None:
            self.model.decoder.load_state_dict(self._original_decoder_state)
        if self._original_prompt_state is not None:
            self.prompt.restore_state(self._original_prompt_state)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to audio features."""
        return self.model.encoder(mel)

    def decode_greedy(self, audio_features: torch.Tensor) -> str:
        """Standard greedy decode (temperature=0) without prompt."""
        result = whisper.decode(
            self.model,
            audio_features.squeeze(0) if audio_features.dim() == 3 else audio_features,
            DecodingOptions(language="en", without_timestamps=True),
        )
        if isinstance(result, list):
            return result[0].text
        return result.text

    def decode_with_prompt_stochastic(
        self,
        audio_features: torch.Tensor,
        temperature: float = 0.5,
        max_tokens: int = 224,
    ) -> dict:
        """Decode with prompt injection and temperature sampling.

        Returns dict with:
            - text: decoded string
            - tokens: token ids [T]
            - log_probs: per-token log probabilities [T]
            - total_log_prob: sum of log probs (scalar)
        """
        model = self.model
        prompt_emb = self.prompt()  # [1, L, D]

        # Get SOT token embedding
        sot_token = torch.tensor(
            [[model.decoder.tokenizer.sot]], device=self.device
        )
        sot_emb = model.decoder.token_embedding(sot_token)  # [1, 1, D]

        # Prepend prompt before SOT
        decoder_input_emb = torch.cat([prompt_emb, sot_emb], dim=1)  # [1, L+1, D]

        tokens = []
        log_probs = []
        all_tokens = [model.decoder.tokenizer.sot]

        for step in range(max_tokens):
            # Build token embeddings for all generated tokens so far
            if step > 0:
                prev_tokens = torch.tensor(
                    [all_tokens[1:]], device=self.device
                )  # exclude SOT already in emb
                prev_emb = model.decoder.token_embedding(prev_tokens)
                current_emb = torch.cat([decoder_input_emb, prev_emb], dim=1)
            else:
                current_emb = decoder_input_emb

            # Add positional encoding - need to handle the prompt offset
            positions = torch.arange(
                current_emb.shape[1], device=self.device
            )
            pos_emb = model.decoder.positional_embedding[positions]
            x = current_emb + pos_emb

            # Run through decoder transformer blocks
            for block in model.decoder.blocks:
                x = block(x, audio_features, mask=model.decoder.mask)

            x = model.decoder.ln(x)
            logits = (
                x[:, -1, :] @ model.decoder.token_embedding.weight.T
            )  # [1, vocab]

            # Temperature-controlled sampling
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            token_id = next_token.item()

            # Compute log prob of chosen token
            log_prob = F.log_softmax(logits, dim=-1)  # use unscaled for RL
            token_log_prob = log_prob[0, token_id].item()

            # Check for EOT
            if token_id == model.decoder.tokenizer.eot:
                break

            tokens.append(token_id)
            log_probs.append(token_log_prob)
            all_tokens.append(token_id)

        text = model.decoder.tokenizer.decode(tokens)
        log_probs_tensor = torch.tensor(log_probs, device=self.device)

        return {
            "text": text.strip(),
            "tokens": torch.tensor(tokens, device=self.device),
            "log_probs": log_probs_tensor,
            "total_log_prob": log_probs_tensor.sum(),
        }

    def generate_candidates(
        self,
        audio_features: torch.Tensor,
        n_candidates: int = 4,
        temp_range: tuple = (0.4, 0.6),
        max_tokens: int = 224,
    ) -> list[dict]:
        """Generate multiple stochastic transcription candidates."""
        candidates = []
        for _ in range(n_candidates):
            temp = torch.empty(1).uniform_(*temp_range).item()
            result = self.decode_with_prompt_stochastic(
                audio_features, temperature=temp, max_tokens=max_tokens
            )
            result["temperature"] = temp
            candidates.append(result)
        return candidates

    def get_trainable_params(
        self, finetune_decoder: bool = True
    ) -> list[dict]:
        """Return parameter groups with appropriate learning rates."""
        param_groups = [
            {"params": [self.prompt.prompt], "lr_scale": 100.0},
        ]
        if finetune_decoder:
            param_groups.append(
                {
                    "params": list(self.model.decoder.parameters()),
                    "lr_scale": 1.0,
                },
            )
        return param_groups
