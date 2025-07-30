from pprint import pprint

from langgraph.prebuilt import create_react_agent

from common.commom_llm import zhipu_model


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model=zhipu_model,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
res=agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
# 美化打印结果
pprint(res)