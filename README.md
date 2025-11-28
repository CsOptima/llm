# Llama 3 Chat API Service

Сервис для генерации текста и чата на базе модели `ruslandev/llama-3-8b-gpt-4o-ru1.0` (или любой другой Llama-3 compatible модели). Реализован на **FastAPI** с поддержкой асинхронного **стриминга** (SSE).

Оптимизирован для работы на **Apple Silicon (M1/M2/M3)** через MPS, а также поддерживает NVIDIA CUDA.

## Требования
- Python 3.10+
- Минимум 16 ГБ RAM (для запуска в float16/bfloat16 на Mac)

## Установка и запуск (Mac M1/M2/M3)

На Mac Docker работает медленно с ML, поэтому рекомендуется запускать локально.

1. **Создайте виртуальное окружение:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
Установите зависимости:

```Bash
pip install fastapi "uvicorn[standard]" transformers torch accelerate bitsandbytes pydantic-settings
```
Запустите сервер:

```Bash

python main.py
```
При первом запуске будет скачана модель (~15 ГБ).

Конфигурация
Настройки задаются в файле .env или переменных окружения:

Переменная	Значение по умолчанию	Описание
MODEL_ID	ruslandev/llama-3-8b-gpt-4o-ru1.0	ID модели на HuggingFace
USE_4BIT	True	Использовать квантование (только для CUDA). На Mac игнорируется.
DEVICE	auto	Автовыбор (mps для Mac, cuda для Nvidia, cpu)
API Endpoints
1. Генерация ответа (Chat Completions)
Полная совместимость с форматом сообщений.
```
URL: /v1/chat/completions
Method: POST
Content-Type: application/json
```
Параметры запроса (Body):

| Поле        | Тип          | Обязательно | Default | Описание                                                   |
| ----------- | ------------ | ----------- |---------|------------------------------------------------------------|
| messages    | list[object] | да          | -       | История диалога.  Объект: {"role": "user/system", "content": "text"}
| max_tokens  | int          | нет         | 512     | Макс. кол-во новых токенов
| temperature | float        | нет         | 0.6     | Креативность (0.0 - 2.0)
| top_p       | float        | нет         | 0.9     | Сэмплирование ядра
| stream      | bool         | нет         | false   | Если true, возвращает поток данных (SSE)

Пример запроса (Обычный):
```Bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "system", "content": "Ты полезный помощник."},
             {"role": "user", "content": "Расскажи про космос кратко."}
           ],
           "stream": false
         }'
```
Пример запроса (Streaming):
В режиме стриминга сервер отдает данные чанками по мере генерации (Server-Sent Events).
```Bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [{"role": "user", "content": "Напиши поэму."}],
           "stream": true
         }'
```