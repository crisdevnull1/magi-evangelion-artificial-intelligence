import datetime
import logging
import os
import sys
from typing import Any, Dict

from langchain.llms import OpenAI

from app import prompts
from app.config import Config


class OpenAILanguageModel:
    def __init__(self, api_key: str, model_name: str, temperature: float):
        self.llm = OpenAI(
            openai_api_key=api_key, model_name=model_name, temperature=temperature
        )

    def process(self, prompt: str) -> str:
        return self.llm(prompt)


class QuestionProcessor:
    def __init__(self, llm: OpenAILanguageModel):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def process_question(self, role: str, question: str) -> str:
        prompt = getattr(prompts, f"{role}_brain").format(question=question)
        response = self.llm.process(prompt)
        self.logger.debug(
            "Role: %s, Question: %s, Response: %s", role, question, response
        )
        return response


class SummaryProcessor:
    def __init__(self, llm: OpenAILanguageModel):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def summarize_responses(self, responses: Dict[str, str]) -> str:
        text = "\n".join(responses.values())
        prompt = prompts.summarize_answers.format(text=text)
        summary = self.llm.process(prompt)
        self.logger.debug("Summary: %s", summary)
        return summary


class Logger:
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        os.makedirs(self.log_directory, exist_ok=True)
        self.log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(self.console_handler)

    def write_to_log(
        self, current_date: str, question: str, responses: Dict[str, str], summary: str
    ) -> None:
        log_file = os.path.join(self.log_directory, f"magi-log-{current_date}.txt")
        with open(log_file, "w+") as f:
            f.write(f"Fecha: {current_date}\n")
            f.write(f"Pregunta: {question}\n\n")
            for role, response in responses.items():
                f.write(f"Respuesta {role.title()}: {response}\n\n")
            f.write(f"Resumen: {summary}\n")
        self.logger.info("Log file saved: %s", log_file)

    def log_to_console(
        self, current_date: str, question: str, responses: Dict[str, str], summary: str
    ) -> None:
        self.logger.info("Fecha: %s", current_date)
        self.logger.info("Pregunta: %s", question)
        for role, response in responses.items():
            self.logger.info("Respuesta %s: %s", role.title(), response)
        self.logger.info("Resumen: %s", summary)


class MAGI:
    roles = ["scientist", "mother", "woman"]

    def __init__(
        self,
        config: Config,
        question_processor: QuestionProcessor,
        summary_processor: SummaryProcessor,
        logger: Logger,
    ):
        self.config = config
        self.question_processor = question_processor
        self.summary_processor = summary_processor
        self.logger = logger

    def ask(self, question: str) -> str:
        responses = {
            role: self.question_processor.process_question(role, question)
            for role in self.roles
        }
        summary = self.summary_processor.summarize_responses(responses)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger.write_to_log(current_date, question, responses, summary)
        self.logger.log_to_console(current_date, question, responses, summary)

        return summary


def new_magi_ai(config: Config) -> MAGI:
    api_key = config.OPENAI_API_KEY
    model_name = "gpt-4"

    standard_llm = OpenAILanguageModel(api_key, model_name, 0.6)
    summarize_llm = OpenAILanguageModel(api_key, model_name, 0)
    question_processor = QuestionProcessor(llm=standard_llm)
    summary_processor = SummaryProcessor(llm=summarize_llm)
    logger = Logger()

    return MAGI(config, question_processor, summary_processor, logger)
