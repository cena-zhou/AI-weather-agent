import os
from dotenv import load_dotenv
import requests

#引入langchain框架的核心组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#========初始化大模型========

#因为用的是 deepseeK 需要指定base_url

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature = 0
)

#========定义工具=====
#使用@tool 装饰器，langchain会自动帮你生成json schema
@tool
def get_weather(city:str) -> str:
    """查询指定城市 的当前天气和温度，参数city必须是城市 中文名"""
    print(f"底层执行正在查{city}的天气...")
    fake_db = {"北京": "晴，28度", "上海": "多云，31度", "深圳": "雷阵雨，30度"}
    return fake_db.get(city, "未知天气")

#把工具放进列表里
tools = [get_weather]

#======4最新版核心改动：绑定工具========
llm_with_tools = llm.bind_tools(tools)

#=====5 创建对话链======
# 直接使用管道符 | 拼接，极简！
# 用户输入 -> 模型思考（并决定是否调工具） -> 输出结果
from langchain_core.output_parsers import StrOutputParser
prompt = ChatPromptTemplate.from_messages([

    ("system", "你是一个有用的助手，可以调用工具查询天气。回答简洁。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

#这是一个最基础的链
chain = prompt | llm_with_tools | StrOutputParser()



#===========6 跑起来================
print("🤖 LangChain Agent 已启动（输入 'q' 退出）")
while True:
    user_input = input("\n你: ")
    if user_input.lower() == 'q':
        break
    
    # 一行代码搞定调用！
    from langchain_core.messages import HumanMessage, ToolMessage
    messages = [HumanMessage(content=user_input)]

    while True:
        # 1. 模型思考
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # 2. 检查是否要调工具
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                # 执行真实工具
                tool_result = get_weather.invoke(tool_call)
                print(f"👉 [执行工具] 查了 {tool_call['args']}，结果：{tool_result.content}")
                
                # 把结果塞回消息
                messages.append(ToolMessage(
                    content=tool_result.content, 
                    tool_call_id=tool_call["id"]
                ))
            # 继续循环，让模型根据结果回答
        else:
            # 没有工具调用，直接输出最终文字
            print(f"\n🤖 AI: {ai_msg.content}")
            break
