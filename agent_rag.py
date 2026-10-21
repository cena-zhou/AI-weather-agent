import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 从项目根目录读取 .env（不要放在 venv/ 里）
load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit(
        "未找到 DEEPSEEK_API_KEY。请在项目根目录的 .env 中配置，例如：\n"
        "DEEPSEEK_API_KEY=sk-你的密钥"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"

#======读取私有知识======
def load_knowledge():
    try:
        with open("my_info.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "没有找到知识库文件。"
    

# ========== 1：定义真实的工具函数 ==========
def get_weather(city: str) -> dict:
    print(f"正在查询{city}的天气...")
    fake_db = {
        "北京": {"weather": "晴", "temp": 28},
        "上海": {"weather": "多云", "temp": 20},
        "广州": {"weather": "雨", "temp": 19},
        "深圳": {"weather": "晴", "temp": 35},
        "长沙": {"weather": "雷阵雨", "temp": 18},
        "武汉": {"weather": "小雨", "temp": 22},
    }
    return fake_db.get(city, {"weather": "未知", "temp": 25})


def get_current_time() -> dict:
    print("正在查询当前时间...")
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}


# ========== 2：把工具「说明书」交给 AI ==========
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气和温度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

#========核心升级，动态注入知识=========
#每次启动时，把知识库里面的内容拼接到系统提示词里
knowledge_text = load_knowledge()
print("\n【调试】系统读到的知识库内容是：", knowledge_text, "\n")

system_prompt = f"""你是一个专属的个人助手
你可以调用工具查询天气，如果有用户问关于他的个人信息，请严格
根据以下提供的【背景知识】来回答，不要瞎编：

{knowledge_text}
"""




# ========== 3：对话主循环 ==========
messages = [{"role": "system", "content": system_prompt}]

print("🤖 Agent 已启动（输入 'q' 退出）")

while True:
    user_input = input("\n你:")
    if user_input.lower() == "q":
        print("再见！")
        break

    # 把用户的话加入记忆（必须在 while 里面）
    messages.append({"role": "user", "content": user_input})

    # Agent：思考 → 执行工具 → 再思考（最多 3 步）
    for step in range(3):
        print(f"\n---- AI 思考第 {step + 1} 步 ---")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 情况 A：AI 要调工具
        if msg.tool_calls:
            # 必须把「带 tool_calls 的 assistant 消息」原样写入记忆
            messages.append(msg)
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                print(f"👉 [AI 决定调用] {func_name}({args})")

                func = TOOL_FUNCTIONS.get(func_name)
                result = func(**args) if func else {"error": "未知工具"}
                print(f"👈 [工具执行结果] {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            # 工具都跑完后，进入下一步 for，让 AI 根据结果再回答
            continue

        # 情况 B：直接给出最终回答
        print(f"\n🤖 AI: {msg.content}")
        messages.append({"role": "assistant", "content": msg.content})
        break  # 结束本轮对话，回到 while 等下一次输入
    else:
        print("\n⚠️ 达到最大思考步数，本轮结束")
