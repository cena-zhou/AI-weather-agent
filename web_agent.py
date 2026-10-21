import os
import streamlit as st
import requests
from dotenv import load_dotenv

# 引入 LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 本地用 .env；Streamlit Cloud 用 Secrets
load_dotenv()


def get_secret(name: str, default: str = "") -> str:
    """优先读 Streamlit Secrets，其次读环境变量 / .env。"""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


# ================= 1. 网页基础配置 =================
st.set_page_config(page_title="Billy_chou的专属天气助理", page_icon="🌤️")
st.title("🌤️ Billy_chou的专属天气助理")
st.caption("基于 LangChain 和 DeepSeek 开发的智能 Agent")

if not get_secret("DEEPSEEK_API_KEY"):
    st.error(
        "未配置 DEEPSEEK_API_KEY。本地请写在 .env；"
        "Streamlit Cloud 请到 App settings → Secrets 中配置。"
    )
    st.stop()


# ================= 2. 定义大模型和工具 =================
# 用 @st.cache_resource 缓存模型，避免每次交互都重新加载
@st.cache_resource
def get_llm_with_tools():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=get_secret("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的当前天气和温度。参数 city 必须是城市中文名，比如'北京'、'株洲'。"""
        api_key = get_secret("QWEATHER_API_KEY", "")
        api_host = get_secret("QWEATHER_HOST", "devapi.qweather.com")

        try:
            # ================= 第一步：查城市 ID =================
            geo_url = (
                f"https://{api_host}/geo/v2/city/lookup?location={city}&key={api_key}"
            )
            geo_resp = requests.get(geo_url, timeout=5).json()

            if geo_resp.get("code") != "200" or not geo_resp.get("location"):
                return f"找不到叫【{city}】的城市，请检查名字是否拼写正确。"

            location_id = geo_resp["location"][0]["id"]
            city_name = geo_resp["location"][0]["name"]

            # ================= 第二步：查真实天气 =================
            weather_url = (
                f"https://{api_host}/v7/weather/now?location={location_id}&key={api_key}"
            )
            weather_resp = requests.get(weather_url, timeout=5).json()

            if weather_resp.get("code") == "200":
                now = weather_resp["now"]
                return f"当前{city_name}的天气是：{now['text']}，温度 {now['temp']}℃。"
            return (
                f"查到了{city_name}，但天气数据获取失败，"
                f"错误码：{weather_resp.get('code')}"
            )

        except Exception as e:
            return f"网络请求出错：{e}"

    @tool
    def get_weather_forecast(city: str) -> str:
        """查询指定城市未来3天的天气预报。参数 city 必须是城市中文名，比如'北京'。"""
        api_key = get_secret("QWEATHER_API_KEY", "")
        api_host = get_secret("QWEATHER_HOST", "devapi.qweather.com")

        try:
            geo_url = (
                f"https://{api_host}/geo/v2/city/lookup?location={city}&key={api_key}"
            )
            geo_resp = requests.get(geo_url, timeout=5).json()
            if geo_resp.get("code") != "200" or not geo_resp.get("location"):
                return f"找不到叫【{city}】的城市。"

            location_id = geo_resp["location"][0]["id"]
            city_name = geo_resp["location"][0]["name"]

            forecast_url = (
                f"https://{api_host}/v7/weather/3d?location={location_id}&key={api_key}"
            )
            forecast_resp = requests.get(forecast_url, timeout=5).json()

            if forecast_resp.get("code") == "200":
                daily_list = forecast_resp["daily"]
                result_text = f"{city_name}未来3天的天气预报：\n"
                for day in daily_list:
                    result_text += (
                        f"日期: {day['fxDate']}, 天气: {day['textDay']}, "
                        f"气温: {day['tempMin']}℃ ~ {day['tempMax']}℃\n"
                    )
                return result_text
            return f"天气数据获取失败，错误码：{forecast_resp.get('code')}"
        except Exception as e:
            return f"网络请求出错：{e}"

    return llm.bind_tools([get_weather, get_weather_forecast]), {
        "get_weather": get_weather,
        "get_weather_forecast": get_weather_forecast,
    }


llm_with_tools, tool_map = get_llm_with_tools()

# ================= 3. 管理网页的“记忆” =================
# Streamlit 每次刷新都会重置变量，所以要把对话历史存在 session_state 里
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把历史对话画在网页上
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ================= 4. 接收用户输入并处理 =================
if user_input := st.chat_input("请问你想查哪个城市的天气？"):
    
    # 1. 显示用户的话
    with st.chat_message("user"):
        st.write(user_input)
    
    # 2. 准备发给模型的格式
    langchain_messages = [HumanMessage(content=user_input)]
    
    # 3. 显示 AI 思考的动画
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考并查询天气..."):
            
            # 把消息发给模型
            ai_msg = llm_with_tools.invoke(langchain_messages)
            
            # 如果模型要调工具
            if ai_msg.tool_calls:
                # 1. 先把 AI 的请求（包含它想调用的工具）存入消息列表
                langchain_messages.append(ai_msg)
                
                # 2. 遍历所有它想调用的工具
                for tool_call in ai_msg.tool_calls:
                    target_tool = tool_map[tool_call["name"]]
                    tool_result = target_tool.invoke(tool_call)
                    
                    # 3. 把每一个工具的返回结果，按顺序存入消息列表
                    langchain_messages.append(ToolMessage(
                        content=tool_result.content, 
                        tool_call_id=tool_call["id"]
                    ))
                
                # 4. 等所有工具执行完毕，把完整的结果集发给模型，获取最终回复！
                final_msg = llm_with_tools.invoke(langchain_messages)
                st.write(final_msg.content)
                
                # 5. 存入网页记忆
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": final_msg.content})
            else:
                # 不需要调工具，直接回答
                st.write(ai_msg.content)
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": ai_msg.content})