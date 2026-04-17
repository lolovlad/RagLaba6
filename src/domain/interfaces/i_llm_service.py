from abc import ABC, abstractmethod


class ILlmService(ABC):
    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        raise NotImplementedError
