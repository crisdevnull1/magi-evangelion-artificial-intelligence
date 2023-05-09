import datetime
from typing import Any

import inject
from langchain.llms import OpenAI

from app.config import Config
from prompt_templates import prompt_templates as template


class Agent:
    config = inject.instance(Config)

    def __init__(self):
        self.llm = OpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model_name="gpt-4",
            temperature=1,
        )
        self.summarize_llm = OpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model_name="gpt-4",
            temperature=0,
        )

    def process_as_scientist(self, question: str) -> str:
        scientist_prompt = template.scientist_brain.format(question=question)
        scientist_response = self.llm(scientist_prompt)
        return scientist_response

    def process_as_mother(self, question: str) -> str:
        mother_prompt = template.mother_brain.format(question=question)
        mother_response = self.llm(mother_prompt)
        return mother_response

    def process_as_woman(self, question: str) -> str:
        woman_prompt = template.woman_brain.format(question=question)
        woman_response = self.llm(woman_prompt)
        return woman_response

    def summarize_question(self, responses: Any) -> str:
        summary = self.summarize_llm(template.summarize_answers.format(text=responses))
        return summary

    def ask(self, question: str):
        scientist_response = self.process_as_scientist(question)
        mother_response = self.process_as_mother(question)
        woman_response = self.process_as_woman(question)

        summary = self.summarize_question(
            [scientist_response, mother_response, woman_response]
        )

        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"summary_log_{current_date}.txt"

        with open(log_file_name, "w+") as f:
            f.write(f"Fecha: {current_date}\n")
            f.write(f"Pregunta: {question}\n\n")
            f.write(f"Respuesta Cientifica: {scientist_response}\n\n")
            f.write(f"Respuesta Madre: {mother_response}\n\n")
            f.write(f"Respuesta Mujer: {woman_response}\n\n")
            f.write(f"Resumen: {summary}\n")

        return summary
