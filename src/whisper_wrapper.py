"""Whisper model wrapper with prompt injection and stochastic decoding."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import whisper
from whisper.decoding import DecodingOptions
from whisper.tokenizer import get_tokenizer

from src.prompt import LearnablePrompt


class WhisperWithPrompt(torch.nn.Module):
    """Wraps a Whisper model to support learnable decoder prompts
    and temperature-controlled stochastic decoding for TTA."""

    def __init__(
        self,
        model_name: str = "tiny",
        prompt_length: int = 4,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        self.tokenizer = get_tokenizer(self.model.is_multilingual)
        embed_dim = self.model.dims.n_text_state
        self.prompt_length = prompt_length
        self.prompt = LearnablePrompt(length=prompt_length, embed_dim=embed_dim)
        self.prompt = self.prompt.to(device)

        self._original_decoder_state: dict | None = None
        self._original_prompt_state: torch.Tensor | None = None

    def save_state(self) -> None:
        """Save model + prompt state for per-sample restoration."""
        self._original_decoder_state = copy.deepcopy(
            self.model.decoder.state_dict()
        )
        self._original_prompt_state = self.prompt.clone_state()

    def restore_state(self) -> None:
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
        inp = (
            audio_features.squeeze(0)
            if audio_features.dim() == 3
            else audio_features
        )
        result = whisper.decode(
            self.model,
            inp,
            DecodingOptions(language="en", without_timestamps=True),
        )
        if isinstance(result, list):
            return result[0].text
        return result.text

    def _build_decoder_input(
        self, all_token_ids: list[int], prompt_emb: torch.Tensor
    ) -> torch.Tensor:
        """Build decoder input: prompt (no pos emb) + tokens (with pos emb).

        The prompt tokens are position-free — they act as content-only context.
        Real tokens (SOT + generated) get positional embeddings starting at 0,
        preserving Whisper's trained positional expectations.
        """
        model = self.model
        token_ids = torch.tensor([all_token_ids], device=self.device)
        token_emb = model.decoder.token_embedding(token_ids)  # [1, T, D]

        # Add positional embeddings only to real tokens (0-indexed)
        n_tokens = token_emb.shape[1]
        positions = torch.arange(n_tokens, device=self.device)
        token_emb = token_emb + model.decoder.positional_embedding[positions]

        # Prompt has no positional embedding — concat before tokens
        return torch.cat([prompt_emb, token_emb], dim=1)  # [1, L+T, D]

    def _run_decoder(
        self, x: torch.Tensor, audio_features: torch.Tensor
    ) -> torch.Tensor:
        """Run through decoder blocks + layernorm → logits."""
        model = self.model
        for block in model.blocks if hasattr(model, "blocks") else model.decoder.blocks:
            x = block(x, audio_features, mask=model.decoder.mask)
        x = model.decoder.ln(x)
        return x[:, -1, :] @ model.decoder.token_embedding.weight.T

    @torch.no_grad()
    def decode_greedy_with_prompt(self, audio_features: torch.Tensor) -> str:
        """Greedy decode (temperature=0) WITH prompt injection.

        Algorithm 1 step 9: y ← Whisper(s; θ, p).
        """
        prompt_emb = self.prompt()  # [1, L, D]

        tokens: list[int] = []
        all_token_ids = [self.tokenizer.sot]

        for _ in range(224):
            x = self._build_decoder_input(all_token_ids, prompt_emb)
            logits = self._run_decoder(x, audio_features)
            token_id = logits.argmax(dim=-1).item()

            if token_id == self.tokenizer.eot:
                break

            tokens.append(token_id)
            all_token_ids.append(token_id)

        return self.tokenizer.decode(tokens).strip()

    def decode_with_prompt_stochastic(
        self,
        audio_features: torch.Tensor,
        temperature: float = 0.5,
        max_tokens: int = 224,
    ) -> dict:
        """Decode with prompt injection and temperature sampling.

        Returns dict with text, tokens, log_probs, and total_log_prob.
        """
        prompt_emb = self.prompt()  # [1, L, D]

        tokens: list[int] = []
        log_probs: list[torch.Tensor] = []
        all_token_ids = [self.tokenizer.sot]

        for _ in range(max_tokens):
            x = self._build_decoder_input(all_token_ids, prompt_emb)
            logits = self._run_decoder(x, audio_features)

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            token_id = next_token.item()

            # Keep log prob in graph for RL gradients
            log_prob = F.log_softmax(logits, dim=-1)
            token_log_prob = log_prob[0, token_id]

            if token_id == self.tokenizer.eot:
                break

            tokens.append(token_id)
            log_probs.append(token_log_prob)
            all_token_ids.append(token_id)

        text = self.tokenizer.decode(tokens)

        if log_probs:
            log_probs_tensor = torch.stack(log_probs)
        else:
            log_probs_tensor = torch.zeros(1, device=self.device, requires_grad=True)

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
        temp_range: tuple[float, float] = (0.4, 0.6),
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
        param_groups: list[dict] = [
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
