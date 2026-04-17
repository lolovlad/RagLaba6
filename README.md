# Лабораторная №6 — Вариант 7

RAG-система для ответов на вопросы по годовому финансовому отчету компании (10-K) с архитектурой Clean Architecture.

## Что реализовано

- Четкое разделение слоев: `Domain`, `Application`, `Infrastructure`, `Presentation`.
- Загрузка документа через API: `POST /documents/upload`.
- Поддержка источников в формате `.pdf` и `.html`.
- Предобработка с `unstructured`: извлечение текста и таблиц.
- Фильтрация шума: номера страниц, колонтитулы и короткие мусорные строки.
- Индексация в Qdrant и поиск по эмбеддингам.
- Генерация ответа по найденному контексту через Yandex GPT.

## Структура проекта

- `src/domain` — сущности и интерфейсы.
- `src/application` — use case-сервисы (индексация, ответ на вопрос).
- `src/infrastructure` — реализации для `unstructured`, `Qdrant`, `SentenceTransformer`, `Yandex GPT`.
- `src/presentation` — HTTP-контроллеры FastAPI.
- `main.py` — точка входа приложения.
- `indexer.py` — пакетная индексация документов из папки `docs`.

## Настройка

Создайте локальный `.env` из шаблона:

```powershell
Copy-Item .env.example .env
```

### Все переменные `.env`

- `QDRANT_HOST` — хост Qdrant (локально обычно `localhost`, в Docker-сети `qdrant`).
- `QDRANT_PORT` — HTTP-порт Qdrant (`6333`).
- `QDRANT_COLLECTION` — имя коллекции, куда пишутся чанки.
- `QDRANT_TIMEOUT` — таймаут операций с Qdrant в секундах.
- `QDRANT_BATCH_SIZE` — размер батча при upsert в Qdrant.
- `PDF_PARSE_BATCH_PAGES` — сколько страниц PDF читать в одном fallback-батче.
- `DOCLING_MAX_PDF_PAGES` — порог страниц, после которого включается более экономный режим парсинга.
- `EMBEDDING_BATCH_SIZE` — размер батча при генерации эмбеддингов.
- `YANDEX_API_KEY` — API-ключ Yandex Cloud (секрет, не коммитить).
- `YANDEX_FOLDER_ID` — ID каталога Yandex Cloud.
- `YANDEX_LLM_BASE_URL` — endpoint LLM API.
- `YANDEX_MODEL_FALLBACK` — URI модели YandexGPT.
- `DOCS_PATH` — папка с исходными документами.
- `OCR_LANGS` — языки OCR через запятую (например `ru,en`).
- `OCR_USE_GPU` — использовать GPU для OCR (`true/false`).

Важно: `.env` должен оставаться только локальным файлом, в git хранится только `.env.example`.

## Запуск

```bash
docker-compose up --build
```

Или локально:

```bash
poetry install
poetry run uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

## Настройка GPU (PyTorch CUDA 12.8)

Для вашего сценария (Docling OCR + эмбеддинги + большие PDF) рекомендуется запускать на GPU.

На CPU обработка может занимать **очень много времени** (десятки минут на один документ) и чаще приводит к таймаутам/ошибкам памяти.

### 1) Установить CUDA-версию PyTorch

Требуемая версия:

- `torch==2.7.1+cu128`

В проекте есть готовый скрипт для Windows PowerShell:

```powershell
.\scripts\setup_torch_gpu.ps1
```

Скрипт:

- удаляет CPU-вариант torch (если был),
- ставит `torch/torchvision/torchaudio` из индекса CUDA 12.8,
- проверяет, что `torch.cuda.is_available() == True`.

### 2) Проверка GPU вручную

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"
```

Ожидается:

- версия содержит `+cu128`
- `True` для CUDA availability.

### 3) OCR на GPU

В `.env`:

```env
OCR_USE_GPU=true
OCR_LANGS=ru,en
```

После изменения перезапустите сервер:

```powershell
poetry run python start_server.py
```

## Примеры API

Загрузка отчета:

```bash
curl -X POST "http://localhost:8000/documents/upload" -F "file=@C:\path\to\10k.pdf"
```

Вопрос по отчету:

```bash
curl -X POST "http://localhost:8000/qa" -H "Content-Type: application/json" -d "{\"question\":\"What was net income in 2023?\"}"
```
