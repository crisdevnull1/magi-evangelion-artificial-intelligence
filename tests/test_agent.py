import unittest
from unittest.mock import patch

from app.agent import new_magi_ai
from app.config import Config


class TestMAGI(unittest.TestCase):
    @patch("app.agent.OpenAILanguageModel.process", return_value="mock_response")
    @patch(
        "app.agent.SummaryProcessor.summarize_responses", return_value="mock_summary"
    )
    def test_ask(self, mock_summarize_responses, mock_process):
        # Mock configurations
        mock_config = Config()
        mock_config.OPENAI_API_KEY = "mock_api_key"

        magi = new_magi_ai(mock_config)
        question = "What is the weather today?"

        summary = magi.ask(question)

        self.assertEqual(summary, "mock_summary")

        # Check if the `process` method of `OpenAILanguageModel` was called
        self.assertEqual(mock_process.call_count, 3)

        # Check if the `summarize_responses` method of `SummaryProcessor` was called with the right argument
        responses = {
            "scientist": "mock_response",
            "mother": "mock_response",
            "woman": "mock_response",
        }
        mock_summarize_responses.assert_called_with(responses)


if __name__ == "__main__":
    unittest.main()
