from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langgraph.prebuilt import create_react_agent
from common.commom_llm import zhipu_model
from langchain_core.messages import AIMessage, ToolMessage


def transfer_money(amount: float, recipient: str, account_number: str):
    """执行银行转账操作"""
    # 请求人工审批，只传递审批信息
    approval_needed = f"请求转账: 向 {recipient} 转账 {amount} 元，账号 {account_number}"
    response = interrupt(approval_needed)

    # 根据人工干预结果执行相应操作
    if response["type"] == "accept":
        return f"已成功向 {recipient} 转账 {amount} 元。交易完成。"
    else:  # 包括reject和其他未知情况
        return "转账操作已被拒绝。"


# 创建agent相关组件
checkpointer = InMemorySaver()
agent = create_react_agent(
    model=zhipu_model,
    tools=[transfer_money],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}


def get_user_choice():
    """获取用户选择"""
    print("\n请选择操作:")
    print("1. accept (批准)")
    print("2. reject (拒绝)")

    choice = input("请输入选择 (1 或 2): ").strip()
    return "accept" if choice == "1" else "reject"


def find_tool_call_id(chunks):
    """从消息历史中找到最近的工具调用ID"""
    for chunk in reversed(chunks):
        if 'agent' in chunk and 'messages' in chunk['agent']:
            for msg in chunk['agent']['messages']:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    return msg.tool_calls[0]['id']
    return None


def run_with_human_intervention():
    """运行带有人工干预处理的示例"""
    print("=== 银行转账示例 ===")
    input_message = {"messages": [{"role": "user", "content": "请向张三转账5000元，账号123456789"}]}

    while True:
        try:
            chunks = []
            for chunk in agent.stream(input_message, config):
                chunks.append(chunk)
                print("Agent输出:", chunk, "\n")

                # 检查是否有中断
                if "__interrupt__" in chunk:
                    print(f"{chunk['__interrupt__'][0].value}")

                    # 获取用户选择
                    response_type = get_user_choice()
                    print(f"您选择了: {response_type}")

                    # 获取工具调用ID
                    tool_call_id = find_tool_call_id(chunks)
                    if not tool_call_id:
                        print("未能找到工具调用信息")
                        return

                    # 构造工具响应（此处需要优化）
                    tool_result = (f"已成功向 张三 转账 5000.0 元。交易完成。"
                                   if response_type == "accept"
                                   else "转账操作已被拒绝。")

                    # 创建工具消息并继续执行
                    tool_message = ToolMessage(content=tool_result, tool_call_id=tool_call_id)
                    input_message = {"messages": [tool_message]}
                    break

            # 如果没有中断，说明执行完成
            else:
                final_message = chunks[-1]['agent']['messages'][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    print("最终结果:", final_message.content)
                break

        except Exception as e:
            print(f"发生错误: {e}")
            break


if __name__ == "__main__":
    run_with_human_intervention()