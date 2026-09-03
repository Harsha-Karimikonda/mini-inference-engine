import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .log import get_logger

logger = get_logger("backend")


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
        logger.info("loading transformers backend", extra={"model": model_name})
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation.streamers import BaseStreamer
        except ImportError as exc:
            raise RuntimeError("Install the [gpu] extra for TransformersBackend") from exc
        self.torch = torch
        self.BaseStreamer = BaseStreamer

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            target_device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
            target_device = "mps"
            dtype = torch.float16
        else:
            target_device = "cpu"
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(target_device)
        self.model.eval()
        self._lock = asyncio.Lock()
        logger.info("transformers backend ready", extra={"model": model_name, "device": target_device, "dtype": str(dtype)})

    async def generate_batch(self, prompts: list[str], max_tokens: int, queues: list[asyncio.Queue[str | None]]) -> None:
        from threading import Thread

        loop = asyncio.get_running_loop()
        batch_size = len(prompts)

        class BatchedStreamer(self.BaseStreamer):
            def __init__(self, tokenizer, batch_sz, lp, qs, eos_id):
                self.tokenizer = tokenizer
                self.batch_size = batch_sz
                self.loop = lp
                self.queues = qs
                self.first = True
                self.finished = [False] * batch_sz
                self.eos_token_id = eos_id

            def put(self, value):
                if self.first:
                    self.first = False
                    return
                token_ids = value.tolist()
                for i, token_id in enumerate(token_ids):
                    if not self.finished[i]:
                        if token_id == self.eos_token_id:
                            self.finished[i] = True
                            self.loop.call_soon_threadsafe(self.queues[i].put_nowait, None)
                        else:
                            text = self.tokenizer.decode([token_id])
                            self.loop.call_soon_threadsafe(self.queues[i].put_nowait, text)

            def end(self):
                for i in range(self.batch_size):
                    if not self.finished[i]:
                        self.finished[i] = True
                        self.loop.call_soon_threadsafe(self.queues[i].put_nowait, None)

        async with self._lock:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            streamer = BatchedStreamer(self.tokenizer, batch_size, loop, queues, self.tokenizer.eos_token_id)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            try:
                await loop.run_in_executor(None, thread.join)
            finally:
                streamer.end()
                await loop.run_in_executor(None, thread.join)

    async def generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        q: asyncio.Queue[str | None] = asyncio.Queue()
        task = asyncio.create_task(self.generate_batch([prompt], max_tokens, [q]))
        try:
            while True:
                token = await q.get()
                if token is None:
                    break
                yield token
        finally:
            await task
