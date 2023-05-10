import datetime
import logging
from typing import Any

import inject
from langchain.llms import OpenAI

import app.prompts as template
from app.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ResponseProcessor:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def process_response(self, question: str, prompt_template: str) -> str:
        prompt = prompt_template.format(question=question)
        response = self.llm(prompt)
        logger.debug(f"Response: {response}")
        return response

    def process_scientist_response(self, question: str) -> str:
        prompt_template = template.scientist_brain
        return self.process_response(question, prompt_template)

    def process_mother_response(self, question: str) -> str:
        prompt_template = template.mother_brain
        return self.process_response(question, prompt_template)

    def process_woman_response(self, question: str) -> str:
        prompt_template = template.woman_brain
        return self.process_response(question, prompt_template)


class Agent:
    config = inject.instance(Config)

    def __init__(self):
        self.llm = OpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=1,
        )
        self.summarize_llm = OpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0,
        )
        self.response_processor = ResponseProcessor(self.llm)

    def summarize_question(self, responses: Any) -> str:
        summary = self.summarize_llm(template.summarize_answers.format(text=responses))
        return summary

    def ask(self, question: str):
        scientist_response = self.response_processor.process_scientist_response(
            question
        )
        mother_response = self.response_processor.process_mother_response(question)
        woman_response = self.response_processor.process_woman_response(question)

        summary = self.summarize_question(
            [scientist_response, mother_response, woman_response]
        )
        logger.debug(f"Summary: {summary}")
        if self.config.DEBUG:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_name = f"logs/execution_{current_date}.txt"

            with open(log_file_name, "w+") as f:
                f.write(f"Fecha: {current_date}\n")
                f.write(f"Pregunta: {question}\n")
                f.write(f"Respuesta Cientifica: {scientist_response}\n")
                f.write(f"Respuesta Madre: {mother_response}\n")
                f.write(f"Respuesta Mujer: {woman_response}\n")
                f.write(f"Resumen: {summary}\n")

        return summary
