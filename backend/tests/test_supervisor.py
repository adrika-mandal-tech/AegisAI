from unittest.mock import MagicMock

from aegis.agents.supervisor import Supervisor
from aegis.schemas.agent import AgentOutput


class TestSupervisor:
    def test_supervise_returns_agent_output(self, mock_llm: MagicMock):
        s = Supervisor(llm=mock_llm)
        result = s.supervise("test task")
        assert isinstance(result, AgentOutput)
        assert result.success is True

    def test_supervise_empty_task(self, mock_llm: MagicMock):
        s = Supervisor(llm=mock_llm)
        result = s.supervise("")
        assert isinstance(result, AgentOutput)
