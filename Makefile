install:
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

pre-commit:
	pre-commit run --all-files --show-diff-on-failure

test:
	python -m unittest discover
	# coverage report
	# coverage html

clima:
	python main.py ask "¿Cómo es el clima en Santiago durante el Invierno?"

pronostico:
	python main.py ask "¿Qué temperatura hará el 3 de mayo en Santiago?"

create-embeddings:
	python main.py create_embedding data/clima-santiago-chile.docx

load-embeddings:
	python main.py load_embedding data/embeddings/clima-santiago-chile_embeddings.csv
