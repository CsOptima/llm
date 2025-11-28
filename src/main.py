from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import asyncio

from schemas import GenerationRequest, GenerationResponse
from service import llm_engine

executor = ThreadPoolExecutor(max_workers=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загружаем модель Llama...")
    llm_engine.load_model()
    yield
    executor.shutdown()

app = FastAPI(title="LLM → SEO Keywords API", lifespan=lifespan)

def parse_queries(content: str) -> List[str]:
    """Извлекает 5 запросов из ответа модели в формате [1]: запрос"""
    queries = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith("[") and "]:" in line:
            query = line.split("]:", 1)[1].strip()
            if query:
                queries.append(query)
    return queries[:5]  # строго не больше 5

@app.post("/v1/chat/completions", response_model=GenerationResponse)
async def chat_completions(request: GenerationRequest):
    try:
        # Строим промпт на основе request.message (summary)
        system_prompt = "Ты — SEO-специалист. Твоя задача — перевести краткое описание контента в реальные поисковые запросы (keywords). Строго соблюдай формат ответа!"
        user_prompt = f"""
На основе предоставленного краткого содержания сгенерируй ровно 5 поисковых запросов, по которым пользователь может искать эту информацию.

Требования:
1. Включи и информационные (как, почему, инструкция), и коммерческие (цена, купить, заказать) запросы.
2. Фразы должны быть естественными для поиска Google/Yandex.
3. Ответ строго списком вида:
[1]: Текст запроса
[2]: Текст запроса
[3]: Текст запроса
[4]: Текст запроса
[5]: Текст запроса

Входное краткое содержание:
{request.message}
"""

        # messages как list[dict] для llama_cpp
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Фиксированные параметры (можно добавить в Request позже)
        params = {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        # Подготовка и генерация
        prepared_messages = llm_engine.prepare_inputs(messages)
        loop = asyncio.get_event_loop()
        full_content = await loop.run_in_executor(
            executor,
            lambda: "".join(llm_engine.generate_stream(params, prepared_messages))
        )

        queries = parse_queries(full_content)
        if len(queries) != 5:
            raise ValueError(f"Модель вернула {len(queries)} запросов вместо 5. Текст: {full_content}")

        return GenerationResponse(queries=queries)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)