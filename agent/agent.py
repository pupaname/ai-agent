import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver  # 👈 1. 导入记忆保存器
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from langchain.agents import create_agent
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_deepseek import ChatDeepSeek
from rich import print as rprint
import media_tools
load_dotenv(override=True)
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")

# Windows 控制台默认 GBK，emoji 打印会抛 UnicodeEncodeError。
# 下面让编不了的字符降级成 '?', 中文照常显示, 程序不再崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 账号里目前只开通了即梦 4.0；3.x 模型调用会返回 HTTP 401 / 50400 Access Denied。
# 这里明确告诉模型该用哪个，失败时怎么退。
SYSTEM_PROMPT = (
    "你是一个中文助手。需要生成图片时，一律使用 jimeng-v40-generate（即梦 4.0）。"
    "不要使用 jimeng-t2i-v31、jimeng-t2i-v30、jimeng-i2i-v30，这些模型在当前账号下未开通，"
    "调用只会返回 HTTP 401 / code 50400 Access Denied。"
    "如果某个工具返回 401 或 50400，请立刻改用 jimeng-v40-generate 重试，并在回复里说明你换了模型。"
    "生成图片时 width/height 用 1024×1024 或 1536×1536 这类常见尺寸。"
)


def tool_text(content):
    """MCP 工具返回值可能是字符串，也可能是 [{'type': 'text', 'text': ...}] 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def download_images(text, limit=4):
    """从即梦工具返回的 JSON 里取出 image_urls，下载到 output/，返回本地路径列表。"""
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    urls = (payload.get("data") or {}).get("image_urls") or []
    if not urls:
        return []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, url in enumerate(urls[:limit], 1):
        dest = OUTPUT_DIR / f"jimeng_{time.strftime('%Y%m%d_%H%M%S')}_{index}.png"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            saved.append(dest)
        except Exception as exc:
            print(f"\n  [下载失败] {exc}")
    return saved
async def main():
    # 1. 初始化 DeepSeek 模型
    llm = init_chat_model(
    model="deepseek:deepseek-flash",
    # profile={"max_input_tokens":128_000},
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},
)
    # 2. 配置 MCP 服务器
    mcp_client = MultiServerMCPClient({
        "12306-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y","12306-mcp"],
        },
        "fetch": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                r"E:\ai agent"  # 👈 替换为真实路径，作为AI能触及的安全边界
            ]
        },
        "jimeng": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y","jimeng4-mcp",
                ],
                "env": {
                    "JIMENG_ACCESS_KEY":os.getenv("JIMENG_ACCESS_KEY"),
                    "JIMENG_SECRET_KEY":os.getenv("JIMENG_SECRET_KEY"),
                }
        }
    })

    print("⏳ 正在连接 MCP 服务器并获取工具列表...")
    tools = await mcp_client.get_tools()
    print(f"✅ 成功连接！共获取到 {len(tools)} 个工具。")

    # 3. 初始化记忆体
    memory = MemorySaver()

    # 4. 创建 Agent，并注入记忆体 checkpointer
    agent = create_agent(
        llm, 
        tools, 
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,# 👈 2. 把记忆体交给 Agent
    )

    # 5. 开启连续对话循环
    print("\n🤖 助手已就绪！(输入 'quit' 或 'exit' 退出)")
    print("   看图: /img 图片路径, 或直接把图片拖进终端")
    print("   说话: /voice, 回车开始录音, 说完再回车")
    print("   录音文件: /audio 文件路径, 手机录音或微信语音都行")
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 再见！")
            break
        if not user_input.strip():
            continue

        # 👈 5. 图片 / 语音输入
        stripped = user_input.strip()
        if stripped.lower() == "/voice":
            spoken = media_tools.listen()
            if not spoken:
                continue
            message_content = spoken
        elif stripped.lower().startswith("/img"):
            image_path = stripped[4:].strip().strip('"').strip("'")
            if not Path(image_path).is_file():
                print(f"  找不到图片文件：{image_path}")
                continue
            message_content = media_tools.image_message(image_path)
        elif stripped.lower().startswith("/audio"):
            audio_path = stripped[6:].strip().strip('"').strip("'")
            spoken = media_tools.transcribe_file(audio_path)
            if not spoken:
                continue
            message_content = spoken
        elif media_tools.is_image_path(stripped):
            message_content = media_tools.image_message(stripped)
        else:
            message_content = user_input

        # 👈 3. 关键：定义 config，thread_id 用于区分不同的用户/会话
        # 只要 thread_id 相同，Agent 就会自动把这个人的历史记录加载出来
        config = {"configurable": {"thread_id": "user_session_001"}}

        print("🤖 助手思考中...", end="", flush=True)
        
        # 6. 流式执行
        try:
            async for step in agent.astream(
                {"messages": [HumanMessage(content=message_content)]},
                config=config,  # 👈 4. 传入 config
                stream_mode="values"
            ):
                latest_msg = step["messages"][-1]
                
                # 打印工具调用过程
                if latest_msg.type == "ai" and getattr(latest_msg, "tool_calls", None):
                    for tc in latest_msg.tool_calls:
                        print(f"  [调用工具] {tc['name']} 参数: {tc['args']}")
                
                # 工具返回：错误显式打出来，生成成功的图片顺手存到 output/
                elif latest_msg.type == "tool":
                    text = tool_text(latest_msg.content)
                    if '"status": "error"' in text or '"error"' in text:
                        print(f"\n  [工具报错] {text[:500]}")
                    for path in download_images(text):
                        print(f"\n  [图片已保存] {path}")
                
                # 打印最终回答
                elif latest_msg.type == "ai" and latest_msg.content:
                    # 移除多余的思考中提示
                    print(f"\r🤖 助手: {latest_msg.content}")
        except Exception as exc:
            # 单轮出错不再让整个会话退出
            print(f"\n❌ 这一轮出错了: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    asyncio.run(main())