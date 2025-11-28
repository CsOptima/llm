from llama_cpp import Llama
from threading import Thread
from config import settings

class LLMService:
    def __init__(self):
        self.llm = None

    def load_model(self):
        print(f"Loading Llama-3 GGUF via llama.cpp on Metal...")
        # Вам нужно сначала скачать файл модели вручную или через скрипт
        # Например, скачайте файл .gguf c HuggingFace и положите в папку models/
        model_path = "../model/model.gguf"
        # n_gpu_layers=-1 переносит ВСЕ слои на M3 Pro GPU
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # Контекст
            n_gpu_layers=-1,  # ВАЖНО: -1 значит все слои на GPU
            verbose=True
        )
        print("Model loaded!")

    def prepare_inputs(self, messages):
        # llama-cpp сам умеет работать с чат-форматом, тут просто прокидываем
        return messages

    def generate_stream(self, params: dict, messages):
        # Специфичный формат для llama-cpp
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.6),
            top_p=params.get("top_p", 0.9),
            stream=True
        )
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                yield delta['content']

llm_engine = LLMService()