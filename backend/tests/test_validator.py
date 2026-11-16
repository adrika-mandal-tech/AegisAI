from unittest.mock import MagicMock

from aegis.agents.validator import Validator
from aegis.schemas.agent import ValidationResult


class TestValidator:
    def test_validate_returns_result(self, mock_llm: MagicMock):
        v = Validator(llm=mock_llm)
        result = v.validate("test content")
        assert isinstance(result, ValidationResult)

    def test_validate_score_range(self, mock_llm: MagicMock):
        v = Validator(llm=mock_llm)
        result = v.validate("content")
        assert 0.0 <= result.score <= 1.0
