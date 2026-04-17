from abc import ABC, abstractmethod

from src.application.dtos.qa_dto import QARequest, QAResponse


class IRagService(ABC):
    @abstractmethod
    def answer_question(self, request: QARequest) -> QAResponse:
        raise NotImplementedError
