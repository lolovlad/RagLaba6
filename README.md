# RAG-система для анализа финансовых отчётов (Лабораторная работа №6, вариант 7)

Сервис «вопрос-ответ» по годовому финансовому отчёту (форма 10-K) на базе **Retrieval-Augmented Generation (RAG)**.

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| API | FastAPI |
| Векторная БД | Qdrant |
| RAG-пайплайн | LangChain (чанкинг) |
| Парсинг PDF | Docling (текст + таблицы) |
| Эмбеддинги | sentence-transformers (intfloat/multilingual-e5-base) |
| LLM | Yandex GPT API |
| Контейнеризация | Docker, docker-compose |

---

## Архитектура RAG

### Offline (индексация)

1. **Парсинг PDF** (Docling): извлечение основного текста и таблиц из отчёта.
2. **Очистка**: удаление номеров страниц, колонтитулов, повторяющихся заголовков.
3. **Чанкинг** (LangChain `RecursiveCharacterTextSplitter`): размер чанка 800, overlap 150.
4. **Эмбеддинги**: модель `intfloat/multilingual-e5-base`, префикс `passage:` для документов.
5. **Qdrant**: коллекция с метрикой COSINE, в метаданных — `source_file`, `page_number`, тип блока (`text` / `table`).

### Online (запрос)

1. **Эмбеддинг запроса** (та же модель, префикс `query:`).
2. **Поиск** в Qdrant: top_k=5, объединение текста и таблиц.
3. **Промпт**: системное сообщение (роль аналитика) + контекст + вопрос.
4. **Генерация** ответа через Yandex GPT; при недостатке контекста — фраза «В отчете отсутствует достаточная информация для точного ответа.»

---

## Структура проекта

```
rag-financial-report/
├── docs/              # PDF-отчёты (положить сюда 10-K)
│   ├── report.pdf
│   └── export/        # после загрузки/индексации: markdown, таблицы, изображения
│       └── <имя_файла>/
│           ├── <имя>.md      # весь документ в markdown
│           ├── tables/       # table_01.md, table_02.md, table_01.csv, ...
│           └── images/       # image_01.png, image_02.png, ...
├── utils/
│   ├── parser.py      # Docling: парсинг PDF, таблицы, очистка
│   └── prompt_builder.py
├── .env               # Создать из env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── indexer.py         # Индексация в Qdrant
├── main.py            # FastAPI: POST /qa, POST /documents/upload
├── rag_pipeline.py    # FinancialRAG, Yandex GPT
└── README.md
```

---

## Запуск

### 1. Переменные окружения

Скопируйте пример и заполните:

```bash
cp .env .env
```

В `.env` укажите:

- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` (для Docker: хост `qdrant`, порт `6333`).
- `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` — ключ и каталог Yandex Cloud для Yandex GPT.
- `YANDEX_MODEL` — например `yandexgpt-lite`.
- `DOCS_PATH` — путь к папке с PDF (по умолчанию `./docs`).

### 2. Запуск сервисов

```bash
docker-compose up --build
```

Поднимаются:

- **Qdrant** — порт 6333.
- **API** — порт 8000.

### 3. Загрузка документов (индексация)

**Через API (рекомендуется):**

Загрузите PDF через endpoint — файл сохранится в `docs/` и будет проиндексирован в Qdrant:

```bash
curl -X POST "http://localhost:8000/documents/upload" ^
  -H "Accept: application/json" ^
  -F "file=@C:\path\to\report.pdf"
```

Ответ:

```json
{
  "filename": "report.pdf",
  "chunks_indexed": 142,
  "status": "ok"
}
```

**Через скрипт (опционально):**

1. Положите PDF в папку `docs/`.
2. На хосте выполните:

```bash
set QDRANT_HOST=localhost
set QDRANT_PORT=6333
python indexer.py
```

Пересоздать коллекцию с нуля: `python indexer.py --recreate`.

### 4. Проверка API

**Пример запроса (curl):**

```bash
curl -X POST "http://localhost:8000/qa" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Какой был чистый доход компании в 2023 году?\"}"
```

**Пример ответа:**

```json
{
  "answer": "По отчёту за 2023 год чистый доход компании составил ...",
  "sources": [
    {"page": 12, "type": "table"},
    {"page": 45, "type": "text"}
  ]
}
```

---

## Дополнительно

- **Таблицы**: извлекаются Docling и сохраняются в виде текста и markdown; участвуют в чанкинге и поиске.
- **Метаданные**: в ответе API возвращаются `sources` с полями `page` и `type` (text/table).
- **Код**: разнесён по модулям (парсер, промпт, индексер, RAG-пайплайн, API), без сокращений, готов к доработке.

После генерации: положите 10-K в `docs/`, запустите `docker-compose up --build`, выполните `python indexer.py`, проверьте через curl.
