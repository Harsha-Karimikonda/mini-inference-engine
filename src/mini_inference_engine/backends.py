from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator


class Backend(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        raise NotImplementedError


class MockBackend(Backend):
    def __init__(self, delay: float = 0.002):
        self.delay = delay

    async def generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        words = ("mini", "together", "inference", "engine", "stream")
        seed = sum(ord(c) for c in prompt) % len(words)
        for i in range(max_tokens):
            await asyncio.sleep(self.delay)
            yield words[(seed + i) % len(words)] + (" " if i + 1 < max_tokens else "")


class TransformersBackend(Backend):
    def __init__(self, model_name: str, device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the [gpu] extra for TransformersBackend") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device != "auto":
            self.model.to(device)
        elif torch.backends.mps.is_available():
            self.model.to("mps")
        elif torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    async def generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        # Kept intentionally simple; the mock backend is used for deterministic service tests.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for token in text.split():
            await asyncio.sleep(0)
            yield token + " "
