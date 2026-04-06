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
    """Wraps a Whisper model to support:
    - Learnable decoder prompts for RL candidate generation
    - Standard whisper.decode() for baseline and final output
    - State save/restore for per-sample adaptation
    """

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
        """Standard Whisper greedy decode. Used for baseline AND final output.

        Uses Whisper's native pipeline — correct KV cache, masking, etc.
        After an RL step, the updated decoder weights are reflected here.
        """
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

    def decode_with_prompt_stochastic(
        self,
        audio_features: torch.Tensor,
        temperature: float = 0.5,
        max_tokens: int = 224,
    ) -> dict:
        """Stochastic decode with prompt for RL candidate generation.

        This is ONLY used to generate candidates and collect differentiable
        log probs for the RL gradient step. The final output always comes
        from decode_greedy() which uses Whisper's native pipeline.

        The prompt acts as a learned bias on the decoder's hidden state,
        steering candidate generation to explore diverse transcriptions.
        """
        model = self.model
        prompt_emb = self.prompt()  # [1, L, D]

        tokens: list[int] = []
        log_probs: list[torch.Tensor] = []
        all_token_ids = [self.tokenizer.sot]

        for _ in range(max_tokens):
            # Standard Whisper embedding: token + positional
            token_tensor = torch.tensor([all_token_ids], device=self.device)
            x = model.decoder.token_embedding(token_tensor)
            x = x + model.decoder.positional_embedding[: x.shape[1]]
            x = x.to(audio_features.dtype)

            # Prepend prompt (no positional embedding — acts as content bias)
            x = torch.cat([prompt_emb.to(x.dtype), x], dim=1)

            # Run decoder blocks
            for block in model.decoder.blocks:
                x = block(x, audio_features, mask=model.decoder.mask)

            x = model.decoder.ln(x)
            logits = (
                x[:, -1, :] @ model.decoder.token_embedding.weight.to(x.dtype).T
            ).float()

            # Temperature sampling
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            token_id = next_token.item()

            # Differentiable log prob for RL
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
