from unittest.mock import MagicMock

from aegis.agents.researcher import Researcher
from aegis.schemas.agent import ResearchResult


class TestResearcher:
    def test_research_returns_result(self, mock_llm: MagicMock):
        r = Researcher(llm=mock_llm)
        result = r.research("test topic")
        assert isinstance(result, ResearchResult)
