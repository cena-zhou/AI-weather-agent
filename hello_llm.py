import os
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
    base_url="https://api.deepseek.com",  # DeepSeek 的接口地址
)

messages = [
    {"role": "system", "content": "你是一个有用的助手，回答要尽量简短。"}
]
print("🤖 聊天机器人已启动（输入 'q' 退出）")

# 开启死循环，不断接收用户输入
while True:
    user_input = input("\n你:")
    # 退出条件
    if user_input.lower() == "q":
        print("再见！")
        break

    # 把用户的话加入记忆列表
    messages.append({"role": "user", "content": user_input})

    # 把整个记忆列表发给模型
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        print(f"\n❌ 出错了: {e}")
        # 失败时把刚加的 user 消息撤掉，避免脏记忆
        messages.pop()
        continue  # 报错就重新开始循环

    # 把模型的回答也加入记忆列表（非常关键，否则模型会忘记自己刚说过啥）
    messages.append({"role": "assistant", "content": reply})

    print(f"🤖 AI: {reply}")
