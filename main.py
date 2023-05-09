import fire

from app.agent import Agent

if __name__ == "__main__":
    agent = Agent()
    fire.Fire(agent)
