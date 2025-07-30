from pprint import pprint
from langgraph.checkpoint.memory import InMemorySaver
from common.commom_llm import zhipu_model
model = zhipu_model
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"今天{city}晴空万里!"

tools = [get_weather]
# 1.历史记忆
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import InMemorySaver
#
#
#
# checkpointer = InMemorySaver()
#
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[get_weather],
#     checkpointer=checkpointer
# )
#
# # Run the agent
# config = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }
#
# sf_response = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
#     config
# )
# print(sf_response)
#
# # Continue the conversation using the same thread_id
# ny_response = agent.invoke(
#     {"messages": [{"role": "user", "content": "How do you think about new york?"}]},
#     config
# )
# print(ny_response)


# 2.总结记忆
# from langmem.short_term import SummarizationNode
# from langchain_core.messages.utils import count_tokens_approximately
# from langgraph.prebuilt import create_react_agent
# from langgraph.prebuilt.chat_agent_executor import AgentState
#
# from typing import Any
#
# summarization_node = SummarizationNode(
#     # 使用 count_tokens_approximately 函数计算当前消息的总令牌数
#     token_counter=count_tokens_approximately,
#     model=model,
#     # 当令牌数超过设定的 max_tokens 阈值时触发总结
#     max_tokens=384,
#     max_summary_tokens=128,
#     output_messages_key="llm_input_messages",
# )
#
#
# class State(AgentState):
#     # NOTE: we're adding this key to keep track of previous summary information
#     # to make sure we're not summarizing on every LLM call
#     context: dict[str, Any]
#
#
# checkpointer = InMemorySaver()
# agent = create_react_agent(
#     model=model,
#     tools=tools,
#     pre_model_hook=summarization_node,
#     state_schema=State,
#     checkpointer=checkpointer,
# )
#
# config = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }
#
# agent.invoke({"messages": "今天上海的天气怎么样？","context": {"user": "张三"}}, config)
# agent.invoke({"messages": "天气还不错吗？今天适合学习呢。你可以为我详细讲一下中华上下五千年吗（不少于500字）","context": {"user": "张三"}}, config)
# agent.invoke({"messages": "好的，我明白了","context": {"user": "张三"}}, config)
# res = agent.invoke({"messages": "今天聊了啥","context": {"user": "张三"}}, config)
# pprint(res)


# # 3.裁剪消息
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langgraph.prebuilt import create_react_agent

# This function will be called every time before the node that calls LLM

def pre_model_hook(state):
    original_count = count_tokens_approximately(state["messages"])
    print(f"原始消息数量: {len(state['messages'])}, token数: {original_count}")

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )

    trimmed_count = count_tokens_approximately(trimmed_messages)
    print(f"裁剪后消息数量: {len(trimmed_messages)}, token数: {trimmed_count}")

    # 显示消息类型，帮助理解调用来源
    print("消息类型序列:", [msg.type for msg in state["messages"]])
    print("-" * 40)

    return {"llm_input_messages": trimmed_messages}

checkpointer = InMemorySaver()
agent = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)
config = {
    "configurable": {
        "thread_id": "1"
    }
}
agent.invoke({"messages": "我是小猪佩奇，你记住了","context": {"user": "张三"}}, config)
agent.invoke({"messages": "今天上海的天气怎么样？","context": {"user": "张三"}}, config)
agent.invoke({"messages": "天气还不错吗？今天适合学习呢。你可以为我详细讲一下中华上下五千年吗（不少于500字）","context": {"user": "张三"}}, config)
agent.invoke({"messages": "好的，我明白了","context": {"user": "张三"}}, config)
res = agent.invoke({"messages": "今天聊了啥，还有我是谁？","context": {"user": "张三"}}, config)
pprint(res)