import os

import requests

from src.domain.interfaces.i_llm_service import ILlmService


class YandexLlmService(ILlmService):
    def __init__(self) -> None:
        self._api_key = os.getenv("YANDEX_API_KEY")
        self._folder_id = os.getenv("YANDEX_FOLDER_ID")
        self._model = os.getenv("YANDEX_MODEL", "yandexgpt-lite")
        self._url = os.getenv(
            "YANDEX_LLM_URL",
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        )
        if not self._api_key or not self._folder_id:
            raise ValueError("YANDEX_API_KEY and YANDEX_FOLDER_ID must be configured")

    def generate_answer(self, question: str, context: str) -> str:
        prompt = (
            "Ты финансовый аналитик. Отвечай только по контексту из 10-K отчета. "
            "Если данных не хватает, честно скажи, что информации недостаточно.\n\n"
            f"Контекст:\n{context}\n\nВопрос: {question}"
        )
        payload = {
            "modelUri": f"gpt://{self._folder_id}/{self._model}",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": "1200"},
            "messages": [{"role": "user", "text": prompt}],
        }
        headers = {"Authorization": f"Api-Key {self._api_key}", "Content-Type": "application/json"}
        response = requests.post(self._url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        return body["result"]["alternatives"][0]["message"]["text"].strip()
