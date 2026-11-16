from unittest.mock import MagicMock

from aegis.agents.reasoner import Reasoner
from aegis.schemas.agent import AgentOutput


class TestReasoner:
    def test_reason_returns_output(self, mock_llm: MagicMock):
        r = Reasoner(llm=mock_llm)
        result = r.reason("test info")
        assert isinstance(result, AgentOutput)
