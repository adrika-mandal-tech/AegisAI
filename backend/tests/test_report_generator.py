from unittest.mock import MagicMock

from aegis.agents.report_generator import ReportGenerator
from aegis.schemas.agent import ReportOutput


class TestReportGenerator:
    def test_generate_returns_output(self, mock_llm: MagicMock):
        rg = ReportGenerator(llm=mock_llm)
        result = rg.generate({"task": "test"})
        assert isinstance(result, ReportOutput)
