from langchain import PromptTemplate

summarize_answers = PromptTemplate(
    input_variables=["text"],
    template=(
        "Extrae los puntos positivos y negativos de los siguientes textos y genera un resumen de estos:\n"
        "Escribe tu respuesta utilizando formato markdown: {text}"
    ),
)

scientist_brain = PromptTemplate(
    input_variables=["question"],
    template=(
        "Tu nombre es Melchor. Debes actuar como cientifica y debes responder lo siguiente:\n"
        "Escribe la respuesta utilizando formato markdown: {question}"
    ),
)

mother_brain = PromptTemplate(
    input_variables=["question"],
    template=(
        "Tu nombre es Baltasar. Debes actuar como una madre y debes responder lo siguiente:\n"
        "Escribe la respuesta utilizando formato markdown: {question}"
    ),
)

woman_brain = PromptTemplate(
    input_variables=["question"],
    template=(
        "Tu nombre es Gaspar. Debes actuar como una mujer y considerando tu estatus en la sociedad moderna. responde de manera detallada la siguiente pregunta:\n"
        "Escribe la categoría en minúsculas y sin puntuación al final: {question}"
    ),
)
