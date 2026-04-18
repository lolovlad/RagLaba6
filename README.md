# Лабораторная №6 — Вариант 7

RAG-сервис для ответов на вопросы по годовому финансовому отчёту компании (10-K) с архитектурой **Clean Architecture**.

## Что реализовано

- Слои: `Domain`, `Application`, `Infrastructure`, `Presentation`.
- **Парсинг PDF/HTML:** Docling + **Surya OCR** (плагин `docling-surya`; требуется Python **≥ 3.12**).
- Чанкинг и эмбеддинги: `langchain-text-splitters`, `sentence-transformers` (модель `intfloat/multilingual-e5-base`, префиксы `query:` / `passage:`).
- Векторная БД: **Qdrant**, upsert батчами.
- LLM: **Yandex GPT** через REST (`requests`).
- Контекст для LLM формируется с **привязкой к фрагменту** (файл, страница при наличии, тип блока), чтобы ответы можно было опирать на источники.

## HTTP API

| Метод | Путь | Описание |
|--------|------|----------|
| `GET` | `/health` | Проверка живости сервиса (`{"status":"ok"}`). |
| `POST` | `/documents/upload` | Загрузка `.pdf` / `.html`, парсинг, эмбеддинги, запись в Qdrant. |
| `POST` | `/qa` | Вопрос по уже проиндексированному отчёту. |

Примеры:

```bash
curl -s http://127.0.0.1:8001/health
```

```bash
curl -X POST "http://127.0.0.1:8001/documents/upload" -F "file=@C:\path\to\report.pdf"
```

```bash
curl -X POST "http://127.0.0.1:8001/qa" -H "Content-Type: application/json" -d "{\"question\":\"Какой чистая прибыль группы за отчётный год?\",\"top_k\":5}"
```

## Переменные окружения (`.env`)

Создайте файл из шаблона:

```powershell
Copy-Item .env.example .env
```

**Не коммитьте `.env`** в git (в репозитории только `.env.example`). Секреты в истории коммитов блокирует GitHub Push Protection.

### Qdrant

| Переменная | Назначение | По умолчанию в коде |
|------------|------------|---------------------|
| `QDRANT_HOST` | Хост Qdrant | `localhost` |
| `QDRANT_PORT` | HTTP-порт | `6333` |
| `QDRANT_COLLECTION` | Имя коллекции | **`financial_reports_10k`** (если переменная не задана) |
| `QDRANT_TIMEOUT` | Таймаут клиента, сек | `120` |
| `QDRANT_BATCH_SIZE` | Размер батча upsert | `64` |

**Docker:** в compose-файлах для сервиса `api` переменные `QDRANT_HOST=qdrant` и `QDRANT_PORT=6333` задаются **поверх** `.env`, поэтому в файле `.env` можно оставить `localhost` для локальных скриптов.

**Индексация с хоста при поднятом только Qdrant в Docker:**

```powershell
docker compose -f docker-compose.cpu.yml up -d qdrant
$env:QDRANT_HOST="localhost"
$env:QDRANT_PORT="6333"
poetry run python indexer.py
```

Порт `6333` проброшен наружу — `localhost:6333` на машине совпадает с контейнером.

### Парсинг и эмбеддинги

| Переменная | Назначение |
|------------|------------|
| `PDF_PARSE_BATCH_PAGES` | Размер батча страниц во fallback-парсере PDF |
| `DOCLING_MAX_PDF_PAGES` | Порог страниц для экономного режима |
| `EMBEDDING_BATCH_SIZE` | Сколько чанков за один вызов энкодера (батч эмбеддингов) |
| `DOCS_PATH` | Каталог с входными документами |

### OCR (Surya)

| Переменная | Назначение |
|------------|------------|
| `OCR_LANGS` | Языки через запятую, например `ru,en` |
| `OCR_USE_GPU` | `true` / `false` — использовать GPU в настройках Surya |

### Yandex GPT

Используются **`YANDEX_MODEL`** и **`YANDEX_LLM_URL`** (см. `src/infrastructure/services/llm_service.py`).

| Переменная | Назначение |
|------------|------------|
| `YANDEX_API_KEY` | API-ключ (секрет) |
| `YANDEX_FOLDER_ID` | ID каталога |
| `YANDEX_MODEL` | Имя модели, например `yandexgpt-lite` |
| `YANDEX_LLM_URL` | URL completion, по умолчанию `https://llm.api.cloud.yandex.net/foundationModels/v1/completion` |

Загрузка `.env`: один раз при импорте `src/container.py` (дублирования в `vector_store_service` нет).

## Логирование

`config_logging.py` настраивает `logging.basicConfig` и понижает шум от `httpx` / `urllib3`. Импортируется в `main.py` и `indexer.py` — ошибок при старте не вызывает.

## Запуск

### Docker: два варианта Compose

| Файл | Образ API | PyTorch | Назначение |
|------|------------|---------|------------|
| `docker-compose.cpu.yml` | `Dockerfile` (`python:3.12-slim`) | колёса из Poetry (обычно **CPU**) | Разработка, CI, машины без GPU |
| `docker-compose.cuda.yml` | `Dockerfile.cuda` (**NVIDIA CUDA 12.8** runtime) | **`torch==2.7.1+cu128`** (+ torchvision/torchaudio cu128) | Сервер с **NVIDIA GPU** и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

Корневой **`docker-compose.yml`** подключает CPU-стек через `include`, поэтому команда по умолчанию:

```bash
docker compose up --build
```

эквивалентна **`docker compose -f docker-compose.cpu.yml up --build`**.

**GPU-стек (CUDA 12.8 + PyTorch cu128 внутри образа):**

```bash
docker compose -f docker-compose.cuda.yml up --build
```

В compose для GPU у сервиса `api` задано `gpus: all` и `OCR_USE_GPU=true`. На хосте должны быть драйвер NVIDIA и поддержка GPU в Docker (`nvidia-smi` в контейнере:  
`docker compose -f docker-compose.cuda.yml run --rm api nvidia-smi`).

Образ API в обоих случаях на **Python 3.12** (требование `docling-surya`).

### Локально (Poetry)

Требуется Python **≥ 3.12** (для Surya).

```bash
poetry install
poetry run python start_server.py
```

или

```bash
poetry run uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

## GPU (PyTorch CUDA 12.8)

Для Docling + Surya + больших PDF **рекомендуется GPU**. На CPU один документ может обрабатываться **очень долго** (десятки минут) и чаще упираться в память/таймауты.

**В Docker** используйте `docker-compose.cuda.yml` — там PyTorch **уже ставится** в образе (`Dockerfile.cuda`) из индекса `cu128`.

**На Windows-хосте без Docker** можно поставить те же колёса в Poetry-окружение:

```powershell
.\scripts\setup_torch_gpu.ps1
```

Проверка:

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

В `.env`: `OCR_USE_GPU=true`.

## Папка `docs/`

Тяжёлые PDF с ограничениями по лицензии **не следует коммитить**. В `.gitignore` игнорируются `docs/**/*.pdf`. Локально кладите отчёты в `docs/`; в git остаётся `docs/.gitkeep`.

Если PDF уже попали в историю git, удалите их из индекса (файлы останутся на диске):

```powershell
git rm --cached docs/*.pdf
```

## Примеры вопросов к отчёту (после индексации)

Файлы вроде `Сбер_ГО_2024_RUS_removed.pdf` часто **скан** без текстового слоя: извлечь из них текст через `pypdf` нельзя, ответы идут после **OCR**. Ниже — типичные вопросы к **годовому отчёту банка** на русском (подставьте формулировки под ваш проиндексированный фрагмент):

1. Какой **чистый финансовый результат** группы за отчётный год?
2. Какова **доля активов**, приходящаяся на розничный бизнес / корпоративный сегмент?
3. Как изменилась **рентабельность капитала (ROE)** по сравнению с предыдущим годом?
4. Какие **основные риски** перечислены в разделе о факторах риска?
5. Каков **коэффициент достаточности капитала** (Basel III) на отчётную дату?
6. Какова **структура кредитного портфеля** по отраслям или сегментам?
7. Какие **дивидендные выплаты** или рекомендации совета директоров указаны за год?
8. Какие **существенные события после отчётной даты** раскрыты в отчёте?

После `POST /documents/upload` отправьте `POST /qa` с одним из вопросов; в ответе смотрите блок `sources` (файл, страница, тип фрагмента).
