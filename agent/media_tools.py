"""图片 / 语音输入的辅助函数。

图片：把本地图片压成 base64，构造 LangChain 的多模态消息（DeepSeek 的 deepseek-flash
      本身就支持看图，不需要额外的视觉模型）。
语音：用麦克风录音（需要 sounddevice），再用 faster-whisper 本地转文字（需要 faster-whisper）。

依赖（只有语音才需要）：
    pip install sounddevice faster-whisper
"""

import base64
import io
import os
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

# 和 agent.py 一样：GBK 控制台下 emoji 不该把程序带走
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# Whisper 模型直接按文件下载到工作区，绕开 huggingface_hub 的缓存/软链接/版本问题
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper"
MODEL_MIRROR = os.getenv("WHISPER_MIRROR", "https://hf-mirror.com")
MODEL_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "preprocessor_config.json"]

# 新版 faster-whisper 会给 av.open 传 metadata_errors，但当前 PyAV 不认这个参数，
# 直接兼容掉，免得每次转写都 TypeError。
try:
    import av as _av

    _av_open = _av.open

    def _av_open_compat(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _av_open(*args, **kwargs)

    _av.open = _av_open_compat
except Exception:
    pass

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

DEFAULT_QUESTION = "看看这张图片，详细说说里面有什么（主体、风格、文字、颜色）。"


def is_image_path(text: str) -> bool:
    """整行输入是不是一个存在的图片文件路径（支持拖拽进终端）。"""
    text = text.strip().strip('"').strip("'")
    if not text:
        return False
    p = Path(text)
    return p.suffix.lower() in IMG_EXTS and p.is_file()


def _to_data_url(path: str, max_side: int = 1536, quality: int = 85) -> str:
    """把图片缩到最长边 max_side 再转 data URL，避免请求体过大。"""
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    buf = io.BytesIO()
    if img.mode == "L":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_message(path: str, question: str = DEFAULT_QUESTION):
    """返回可以直接放进 HumanMessage 的 content 列表。"""
    path = path.strip().strip('"').strip("'")
    return [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": _to_data_url(path)}},
    ]


def record_wav(path: str, samplerate: int = 16000, max_seconds: int = 60) -> str:
    """按 Enter 开始录音，再按 Enter 结束。需要 sounddevice。"""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("缺少录音依赖，先运行：pip install sounddevice")

    frames = []
    stop = threading.Event()

    def wait_enter():
        input()
        stop.set()

    print("  🎙️  开始说话，说完按 Enter 结束…", flush=True)
    threading.Thread(target=wait_enter, daemon=True).start()
    started = time.time()
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
            while not stop.is_set() and time.time() - started < max_seconds:
                block, _ = stream.read(1024)
                frames.append(block.copy())
    except Exception as exc:
        raise RuntimeError(
            f"打不开麦克风（{exc}）。这台机器可能没有音频输入设备，"
            "可以改用 /audio 发送手机录的音频文件。"
        )

    if not frames:
        return ""
    audio = np.concatenate(frames, axis=0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return path


_MODEL = None


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """带重试的流式下载，边下边显示进度。"""
    import requests

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as resp:
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total and done % (20 << 20) < (1 << 20):
                            print(f"      {dest.name} {done / 1048576:.0f}/{total / 1048576:.0f} MB", flush=True)
            part.replace(dest)
            return True
        except Exception as exc:
            print(f"      下载 {dest.name} 第 {attempt} 次失败: {exc}", flush=True)
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            time.sleep(2)
    return False


def ensure_local_model(size: str = "base") -> str:
    """把 Systran/faster-whisper-<size> 的模型文件下到 models/whisper/<size>。"""
    repo = f"Systran/faster-whisper-{size}"
    dest = MODEL_DIR / size
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "model.bin").exists() and (dest / "config.json").exists():
        return str(dest)
    print(f"  ⏳ 首次使用：下载 Whisper 模型 {size}（约几百 MB，只下一次）…", flush=True)
    for name in MODEL_FILES:
        target = dest / name
        if target.exists():
            continue
        _download_file(f"{MODEL_MIRROR}/{repo}/resolve/main/{name}", target)
    if not (dest / "model.bin").exists():
        raise RuntimeError(
            f"模型下载失败。可以手动下载 {MODEL_MIRROR}/{repo} 里的 "
            f"config.json / model.bin / tokenizer.json / vocabulary.txt 到 {dest}"
        )
    return str(dest)


def _whisper_model(size: str | None = None):
    global _MODEL
    size = size or os.getenv("WHISPER_MODEL", "base")
    if _MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("缺少转写依赖，先运行：pip install faster-whisper")
        local_dir = ensure_local_model(size)
        print(f"  ⏳ 加载模型 {size}（第一次会慢一点）…", flush=True)
        _MODEL = WhisperModel(local_dir, device="cpu", compute_type="int8")
    return _MODEL


def transcribe(path: str, language: str = "zh") -> str:
    """本地 faster-whisper 转文字（繁体自动转简体）。"""
    model = _whisper_model()
    segments, _info = model.transcribe(path, language=language, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    try:  # Whisper 的中文经常输出繁体，转成简体更顺手
        import warnings

        warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
        from zhconv import convert

        text = convert(text, "zh-cn")
    except Exception:
        pass
    return text


def listen(out_dir: str | None = None) -> str:
    """录一段话并转成文字，失败时返回空字符串。"""
    out_dir = out_dir or str(Path(__file__).resolve().parent.parent / "output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wav = str(Path(out_dir) / f"voice_{time.strftime('%Y%m%d_%H%M%S')}.wav")
    try:
        record_wav(wav)
        text = transcribe(wav)
    except RuntimeError as exc:
        print(f"  [语音不可用] {exc}")
        return ""
    print(f"  📝 识别结果：{text or '（没听清）'}")
    return text


def transcribe_file(path: str) -> str:
    """把已有的录音文件转成文字（手机录音、微信语音等）。"""
    path = path.strip().strip('"').strip("'")
    if not Path(path).is_file():
        print(f"  找不到音频文件：{path}")
        return ""
    try:
        text = transcribe(path)
    except RuntimeError as exc:
        print(f"  [转写不可用] {exc}")
        return ""
    print(f"  📝 识别结果：{text or '（没听清）'}")
    return text
