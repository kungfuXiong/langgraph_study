from langgraph.prebuilt import create_react_agent
from common.commom_llm import zhipu_model


def tool() -> None:
    """Testing tool."""
    ...

agent = create_react_agent(
    zhipu_model,
    tools=[tool],
)

try:
    mermaid_code = agent.get_graph().draw_mermaid_png( )
    with open("graph.jpg", "wb") as f:
        f.write(mermaid_code)
except Exception as err:
    # This requires some extra dependencies and is optional
    print(err)
