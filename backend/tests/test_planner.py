from unittest.mock import MagicMock

from aegis.agents.planner import Planner
from aegis.schemas.agent import AgentPlan


class TestPlanner:
    def test_plan_returns_agent_plan(self, mock_llm: MagicMock):
        p = Planner(llm=mock_llm)
        result = p.plan("test task")
        assert isinstance(result, AgentPlan)

    def test_plan_empty_task(self, mock_llm: MagicMock):
        p = Planner(llm=mock_llm)
        result = p.plan("")
        assert isinstance(result, AgentPlan)
