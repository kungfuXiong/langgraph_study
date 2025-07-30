from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from common.commom_llm import zhipu_model

# 初始化语言模型
llm = zhipu_model


# 1. 定义图的状态结构
class State(TypedDict):
    """
    定义图中传递的状态结构
    messages: 消息列表，使用add_messages函数来处理消息的添加逻辑
    """
    # （在这种情况下，它将消息追加到列表中，而不是覆盖它们）
    messages: Annotated[list, add_messages]


# 创建状态图构建器
graph_builder = StateGraph(State)


def chatbot(state: State):
    """
    聊天机器人节点函数
    接收当前状态，调用语言模型处理消息，并返回新的消息

    Args:
        state: 当前图状态，包含消息历史

    Returns:
        包含新消息的字典
    """
    return {"messages": [llm.invoke(state["messages"])]}


# 添加聊天机器人节点到图中
# 第一个参数是唯一的节点名称
# 第二个参数是节点被使用时将调用的函数或对象
graph_builder.add_node("chatbot", chatbot)

# 添加从起始节点到聊天机器人节点的边
graph_builder.add_edge(START, "chatbot")

# 编译图以准备执行
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
    流式处理图的更新并打印助手回复

    Args:
        user_input: 用户输入的内容
    """
    # 流式执行图，传入用户消息
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # 遍历每个事件的值并打印助手的回复
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# 主循环：接收用户输入并获取助手回复
while True:
    try:
        user_input = input("User: ")
        # 检查用户是否想要退出
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # 处理用户输入并流式输出回复
        stream_graph_updates(user_input)
    except:
        # 如果input()不可用则使用默认输入作为后备
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
