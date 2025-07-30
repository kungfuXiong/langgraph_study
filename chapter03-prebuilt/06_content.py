from langchain_core.tools import tool
from pprint import pprint

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from common.commom_llm import zhipu_model

"""

上下文管理
"""


# 1.配置 (静态上下文)
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[]
# )
# res=agent.invoke(
#     {"messages": [{"role": "user", "content": "hi!"}]},
#     config={"configurable": {"user_id": "user_123"}}
# )
# pprint(res)

# 2.状态 (动态上下文)
# class CustomerServiceState(AgentState):
#     user_name: str
#     order_id: str
#     customer_level: str  # VIP/普通用户
#
# @tool
# def get_order_info(order_id: str):
#     """Get order information."""
#     return f"Order {order_id} is expected to arrive in 5 days."
#
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[get_order_info],
#     state_schema=CustomerServiceState,
# )
#
# # 实际调用时
# res = agent.invoke({
#     "messages": "我的订单什么时候能到？",
#     "user_name": "张三",
#     "order_id": "ORD20230405001",
#     "customer_level": "VIP"
# })
# pprint(res)

# 3.使用上下文自定义提示：
# 3.1使用配置
# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"
#
#
# def prompt(
#         state: AgentState,
#         config: RunnableConfig,
# ) -> list[AnyMessage]:
#     user_name = config["configurable"].get("user_name")
#     system_msg = f"You are a helpful assistant. User's name is {user_name}"
#     return [{"role": "system", "content": system_msg}] + state["messages"]
#
#
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[get_weather],
#     prompt=prompt
# )
#
# res = agent.invoke(
#     {"messages":"我是谁？"},
#     config={"configurable": {"user_name": "John Smith"}}
# )
#
# pprint(res)

# 3.使用上下文自定义提示：
# 3.1使用状态
# class CustomState(AgentState):
#     user_name: str
#
# def prompt(
#     state: CustomState
# ) -> list[AnyMessage]:
#     user_name = state["user_name"]
#     system_msg = f"You are a helpful assistant. User's name is {user_name}"
#     return [{"role": "system", "content": system_msg}] + state["messages"]
#
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[],
#     state_schema=CustomState,
#     prompt=prompt
# )
#
# res=agent.invoke({
#     "messages": "hi!",
#     "user_name": "John Smith"
# })
# pprint(res)

# 4.工具中使用上下文：
# def get_user_info(
#     config: RunnableConfig,
# ) -> str:
#     """Look up user info."""
#     user_id = config["configurable"].get("user_id")
#     return "User is John Smith" if user_id == "John Smith" else "Unknown user"
#
# agent = create_react_agent(
#     model=zhipu_model,
#     tools=[get_user_info],
# )
#
# res=agent.invoke(
#     {"messages": [{"role": "user", "content": "look up user information"}]},
#     config={"configurable": {"user_id": "John Smith"}}
# )
# pprint(res)

# 使用 Annotated 类型注解，表明 state 参数是 CustomState 类型，并通过 InjectedState 注入
from typing import Annotated
from langgraph.prebuilt import InjectedState
class CustomState(AgentState):
    user_id: str

def get_user_info(
    state: Annotated[CustomState, InjectedState("user_id")]
) -> str:
    """Look up user info."""
    user_id = state["user_id"]
    return "User is John Smith" if user_id == "John Smith" else "Unknown user"

agent = create_react_agent(
    model=zhipu_model,
    tools=[get_user_info],
    state_schema=CustomState,
)

res=agent.invoke({
    "messages": "look up user information",
    "user_id": "John Smith"
})
pprint(res)