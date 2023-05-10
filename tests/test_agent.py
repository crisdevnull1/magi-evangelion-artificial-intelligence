import unittest
from unittest.mock import MagicMock

from app.agent import Agent, ResponseProcessor


class TestResponseProcessor(unittest.TestCase):
    def test_process_scientist_response(self):
        llm_mock = MagicMock()
        llm_mock.return_value = "respuesta de científico"
        processor = ResponseProcessor(llm=llm_mock)

        response = processor.process_scientist_response("pregunta de ejemplo")
        llm_mock.assert_called_once()
        self.assertEqual(response, "respuesta de científico")

    def test_process_mother_response(self):
        llm_mock = MagicMock()
        llm_mock.return_value = "respuesta de madre"
        processor = ResponseProcessor(llm=llm_mock)

        response = processor.process_mother_response("pregunta de ejemplo")
        llm_mock.assert_called_once()
        self.assertEqual(response, "respuesta de madre")

    def test_process_woman_response(self):
        llm_mock = MagicMock()
        llm_mock.return_value = "respuesta de mujer"
        processor = ResponseProcessor(llm=llm_mock)

        response = processor.process_woman_response("pregunta de ejemplo")
        llm_mock.assert_called_once()
        self.assertEqual(response, "respuesta de mujer")

    def test_ask(self):
        agent = Agent()
        agent.response_processor.process_scientist_response = MagicMock(
            return_value="respuesta de científico"
        )
        agent.response_processor.process_mother_response = MagicMock(
            return_value="respuesta de madre"
        )
        agent.response_processor.process_woman_response = MagicMock(
            return_value="respuesta de mujer"
        )
        agent.summarize_llm = MagicMock(return_value="resumen de respuestas")

        summary = agent.ask("pregunta de ejemplo")

        agent.response_processor.process_scientist_response.assert_called_once()
        agent.response_processor.process_mother_response.assert_called_once()
        agent.response_processor.process_woman_response.assert_called_once()
        agent.summarize_llm.assert_called_once()

        self.assertEqual(summary, "resumen de respuestas")


if __name__ == "__main__":
    unittest.main()
