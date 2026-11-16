from unittest.mock import MagicMock

from aegis.agents.critic import Critic
from aegis.schemas.agent import CritiqueResult


class TestCritic:
    def test_critique_returns_result(self, mock_llm: MagicMock):
        c = Critic(llm=mock_llm)
        result = c.critique("test content")
        assert isinstance(result, CritiqueResult)

    def test_critique_empty_content(self, mock_llm: MagicMock):
        c = Critic(llm=mock_llm)
        result = c.critique("")
        assert isinstance(result, CritiqueResult)
