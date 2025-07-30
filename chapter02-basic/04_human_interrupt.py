from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

from common.commom_llm import zhipu_model

llm = zhipu_model


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """专业的高级客服人员，可以提供运营商疑难杂症的解决方案"""
    human_response = interrupt({"query": query})
    return human_response["data"]


tool = TavilySearch(max_results=2, tavily_api_key='')
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
    except Exception as e:
        # 当 human_assistance 工具被调用时会抛出中断异常
        # 这里我们捕获中断并提供人工响应
        print("-----------------------------------人工干预------------------------------------------")

        # 等待人工输入
        expert_input = input("expert: ")
        # 提供人工响应，作为模型参考
        human_response = (expert_input)

        # 创建恢复命令，无需传入上下文，state会管理
        human_command = Command(resume={"data": human_response})

        # 使用恢复命令继续执行图
        try:
            for event in graph.stream(human_command, config):
                for value in event.values():
                    print("Assistant:", value["messages"][-1].content)
        except Exception as resume_error:
            print(f"Error during resume: {resume_error}")


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)
