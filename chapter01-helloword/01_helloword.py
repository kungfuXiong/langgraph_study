from pprint import pprint

from langgraph.prebuilt import create_react_agent

from common.commom_llm import zhipu_model


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}天气今天很不错!"


agent = create_react_agent(
    model=zhipu_model,
    tools=[get_weather],
    prompt="你是一个得力助手！"
)

# Run the agent
res=agent.invoke(
    {"messages": [{"role": "user", "content": "南京今天天气如何"}]}
)
# 美化打印结果
pprint(res)