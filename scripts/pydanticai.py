from pydantic_ai import Agent

agent = Agent(name="assistant", system_prompt="You're a helpful assistant.")

print(agent.run_sync("What is the capital of France?"))