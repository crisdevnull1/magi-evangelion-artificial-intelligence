import os

import inject

try:
    # Try to load the .env.local file if it exists
    from dotenv import load_dotenv

    dotenv_path = ".env.local"
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    # Ignore the ImportError if the dotenv library is not installed
    pass


class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def configure(binder):
    binder.bind(Config, Config())


inject.configure(configure)
