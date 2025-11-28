from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from schemas import GenerationRequest, GenerationResponse
from service import llm_engine


# 1. Lifespan: Загрузка модели ПЕРЕД стартом приема запросов
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код при запуске
    try:
        llm_engine.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    yield
    # Код при выключении (если нужно освободить VRAM)
    # del llm_engine.model
    # torch.cuda.empty_cache()


app = FastAPI(title="Llama 3 API", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def generate_chat(request: GenerationRequest):
    """
    Основной эндпоинт. Поддерживает как stream=True, так и обычный ответ.
    """
    try:
        # Преобразуем Pydantic модели в список dict
        messages_dict = [m.model_dump() for m in request.messages]
        input_ids = llm_engine.prepare_inputs(messages_dict)

        gen_params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }

        # Вариант 1: Потоковый ответ (рекомендуется для LLM)
        if request.stream:
            return StreamingResponse(
                llm_engine.generate_stream(gen_params, input_ids),
                media_type="text/event-stream"
            )

        # Вариант 2: Обычный ответ (ждем конца генерации)
        else:
            full_response = ""
            # Используем тот же генератор, но собираем строку целиком
            for chunk in llm_engine.generate_stream(gen_params, input_ids):
                full_response += chunk

            return GenerationResponse(content=full_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
