import logging

import fire

from app.agent import new_magi_ai
from app.config import Config


def main():
    logging.basicConfig(level=logging.DEBUG)
    config = Config()
    agent = new_magi_ai(config)
    fire.Fire(agent)


if __name__ == "__main__":
    main()
