import json
import requests
import asyncio
import re
import base64
import os
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import PushAudioOutputStream
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import io
import wave
from faster_whisper import WhisperModel
import tempfile
import uvicorn
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1) 加载模型 
#    - model_size: "tiny", "base", "small", "medium", "large"
#    - device: "cpu" 或 "cuda"（如有 GPU）
#    - compute_type: "int8_float16"（速度快，精度略低）或 "float32"（默认高精度）
whisper_model = WhisperModel(
    "large", 
    device="cpu", 
    compute_type="int8"#float32高精度

)

# 配置日志
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 挂载静态文件
app.mount("/static", StaticFiles(directory="web_demo/static"), name="static")

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "southeastasia")

# 存储WebSocket连接的管理类
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(f"WebSocket连接已接受，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

manager = ConnectionManager()

def wrap_text_with_lang(text):
    """
    检测文本是否为英文，并相应地添加SSML标签
    如果是纯英文文本，用<lang>标签包裹指定为英语
    否则保持原样，作为中文处理
    """
    # 如果文本为空，返回默认文本
    if not text or text.strip() == "":
        return "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>空文本</speak>"
    
    # 检查传入的文本是否已经包含SSML标签，如果有则直接返回
    if text.strip().startswith("<speak") and text.strip().endswith("</speak>"):
        return text
    
    # 处理特殊符号 - 先转义所有XML特殊字符
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")
    
    # 检查是否为纯英文文本（字母、数字、标点和空格）
    is_pure_english = all(c.isascii() and (c.isalpha() or c.isdigit() or c.isspace() or not c.isalnum()) for c in text)
    
    # 检查是否为长文本 - 超过100个字符的英文文本可能需要特殊处理
    if is_pure_english:
        # 对于英文长文本，直接使用英文语音，不使用lang标签
        if len(text) > 100:
            # 直接使用英文模式，不使用lang标签
            return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{escaped_text}</speak>"
        else:
            # 短英文文本，使用lang标签包裹
            return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'><lang xml:lang='en-US'>{escaped_text}</lang></speak>"
    
    # 如果包含中文和英文混合
    if any(ord(c) > 127 for c in text):
        # 对于混合文本，简化处理，避免复杂的SSML标签嵌套
        # 简单地将整个文本作为中文处理
        return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>{escaped_text}</speak>"
    
    # 默认作为中文处理
    return f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>{escaped_text}</speak>"

def get_audio(text, voice_speed, voice_id):
    """
    使用Azure语音服务将文本转换为语音，保持统一的音频格式以支持唇形同步
    """
    logger.debug(f"生成语音: text={text}, voice_speed={voice_speed}, voice_id={voice_id}")
    try:
        # 配置Azure语音服务
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        
        # 设置语音合成选项
        if voice_id:
            # 映射内部voice_id到Azure的语音ID
            voice_mapping = {
                "male-qn-qingse": "zh-CN-YunxiNeural",  # 青涩男声
                "male-qn-badao": "zh-CN-YunjianNeural",  # 霸气男声
                "wumei_yujie": "zh-CN-XiaoxiaoNeural",  # 妩媚女声
                "female-tianmei": "zh-CN-XiaoyiNeural"   # 甜美女声
            }
            
            # 中文语音到英文语音的性别映射
            english_voice_mapping = {
                "zh-CN-YunxiNeural": "en-US-GuyNeural",       # 云希(男) -> Guy(男)
                "zh-CN-YunjianNeural": "en-US-JasonNeural",   # 云健(男) -> Jason(男)
                "zh-CN-XiaoxiaoNeural": "en-US-JennyNeural",  # 晓晓(女) -> Jenny(女)
                "zh-CN-XiaoyiNeural": "en-US-AriaNeural"      # 晓伊(女) -> Aria(女)
            }
            
            # 检测是否为纯英文文本，如果是则使用英文语音
            is_pure_english = all(c.isascii() and (c.isalpha() or c.isdigit() or c.isspace() or not c.isalnum()) for c in text)
            
            # 先获取对应的中文语音
            chinese_voice = voice_mapping.get(voice_id, "zh-CN-YunxiNeural")
            
            if is_pure_english and len(text) > 100:
                # 对于长英文文本，使用对应性别的英文语音
                azure_voice = english_voice_mapping.get(chinese_voice, "en-US-GuyNeural")
                logger.debug(f"检测到长英文文本，从{chinese_voice}切换到英文语音: {azure_voice}")
            else:
                azure_voice = chinese_voice
        else:
            # 默认中文语音
            chinese_voice = "zh-CN-XiaoxiaoNeural"  # 默认使用女生
            
            # 检测是否为纯英文文本，如果是则使用英文语音
            is_pure_english = all(c.isascii() and (c.isalpha() or c.isdigit() or c.isspace() or not c.isalnum()) for c in text)
            if is_pure_english and len(text) > 100:
                # 对于长英文文本，使用默认英文女声
                azure_voice = "en-US-JennyNeural"  # 使用英文女声
                logger.debug(f"检测到长英文文本，切换到英文语音: {azure_voice}")
            else:
                azure_voice = chinese_voice
            
        logger.debug(f"使用Azure语音: {azure_voice}")
        speech_config.speech_synthesis_voice_name = azure_voice
        
        # 设置语速
        if voice_speed:
            try:
                speed = float(voice_speed)
                if 0.5 <= speed <= 2.0:  # Azure接受0.5到2.0之间的语速
                    rate_setting = f"{int((speed-1)*100):+d}%"
                    logger.debug(f"设置语速: {rate_setting}")
                    speech_config.speech_synthesis_rate = rate_setting
            except Exception as e:
                logger.warning(f"设置语速失败: {e}")
                pass
        
        # 设置音频输出格式为16kHz, 16bit PCM - 这对唇形同步非常重要
        output_format = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        logger.debug(f"设置输出格式: {output_format}")
        speech_config.set_speech_synthesis_output_format(output_format)

        # 推式流，用来接收 SDK 合成的数据
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        # 将文本处理为SSML格式，以支持中英文自动切换
        ssml_text = wrap_text_with_lang(text)
        logger.debug(f"使用SSML: {ssml_text}")
        
        # 检查文本是否为空或只包含空格
        if text.strip() == "":
            logger.warning("检测到空文本，跳过语音合成")
            silent_audio = create_silent_audio(500)  # 500ms的静音
            base64_string = base64.b64encode(silent_audio).decode('utf-8')
            return base64_string
            
        # 对于长文本，尝试简单合成
        if len(text) > 500:
            logger.debug("检测到长文本，使用简单文本合成而非SSML")
            result = speech_synthesizer.speak_text_async(text).get()
        else:
            # 使用SSML进行语音合成
            logger.debug("开始合成语音...")
            result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # 从流中获取音频数据
            audio_data = result.audio_data
            audio_size = len(audio_data)
            logger.debug(f"语音合成成功: 音频大小={audio_size} 字节")
            
            # 编码为base64
            base64_string = base64.b64encode(audio_data).decode('utf-8')
            logger.debug(f"音频已转换为Base64，长度={len(base64_string)}")
            return base64_string
        else:
            # 如果合成失败
            error_msg = f"语音合成失败: {result.reason}"
            logger.error(error_msg)
            
            # 尝试使用简单合成
            logger.debug("尝试使用简单文本合成")
            try:
                simple_result = speech_synthesizer.speak_text_async(text).get()
                if simple_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_data = simple_result.audio_data
                    base64_string = base64.b64encode(audio_data).decode('utf-8')
                    return base64_string
            except Exception as e:
                logger.error(f"简单合成也失败: {e}")
            
            # 创建一个静音的WAV文件
            silent_audio = create_silent_audio(500)  # 500ms的静音
            base64_string = base64.b64encode(silent_audio).decode('utf-8')
            return base64_string
            
    except Exception as e:
        logger.exception(f"TTS API调用失败: {e}")
        # 出错时生成静音
        silent_audio = create_silent_audio(500)  # 500ms的静音
        base64_string = base64.b64encode(silent_audio).decode('utf-8')
        return base64_string

def create_silent_audio(duration_ms):
    """创建指定时长的静音WAV数据"""
    # 设置音频参数
    channels = 1
    sample_width = 2  # 2字节 = 16位
    framerate = 16000
    n_frames = int(framerate * duration_ms / 1000)
    
    # 创建内存缓冲区
    buffer = io.BytesIO()
    
    # 创建WAV文件
    with wave.open(buffer, 'wb') as wave_write:
        wave_write.setnchannels(channels)
        wave_write.setsampwidth(sample_width)
        wave_write.setframerate(framerate)
        # 生成静音数据 (全0)
        wave_write.writeframes(b'\x00' * n_frames * sample_width * channels)
    
    # 获取WAV数据
    buffer.seek(0)
    return buffer.read()

async def llm_answer(prompt):
    # 调用智谱AI的API获取回答
    logger.debug(f"发送请求到智谱AI: prompt={prompt}")
    try:
        # 导入智谱AI的SDK
        from zhipuai import ZhipuAI
        
        client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        
        # 调用API
        response = client.chat.completions.create(
            model="glm-4",  # 使用glm-4模型
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # 获取回答
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content
            logger.debug(f"智谱AI回答: {answer[:100]}...")  # 只记录前100个字符防止日志过长
            return answer
        else:
            # 如果没有返回预期的回答，提供一个默认回答
            logger.warning("智谱AI没有返回预期的回答")
            return "抱歉，我现在无法回答您的问题。请稍后再试。"
    
    except Exception as e:
        logger.exception(f"调用智谱AI API出错: {e}")
        # 错误时返回默认回答
        return f"抱歉，服务暂时不可用。错误信息: {str(e)}"

def split_sentence(sentence, min_length=30):  # 增加最小长度阈值
    # 定义包括小括号在内的主要标点符号
    punctuations = r'[。？！；…，、()（）]'
    # 使用正则表达式切分句子，保留标点符号
    parts = re.split(f'({punctuations})', sentence)
    parts = [p for p in parts if p]  # 移除空字符串
    sentences = []
    current = ''
    
    # 首先尝试按句号、问号、感叹号等标点分割
    end_punctuations = ['。', '？', '！', '；', '.', '?', '!', ';']
    
    for part in parts:
        current += part
        # 如果当前片段以句子结束标点结尾，且长度足够长，就添加到结果中
        if part in end_punctuations and len(current) >= min_length:
            sentences.append(current)
            current = ''
    
    # 如果没有用句号等分成足够的句子，尝试用长度来分割
    if len(sentences) <= 1 and len(sentence) > min_length * 3:
        current = ''
        for part in parts:
            if len(current) + len(part) >= min_length * 2:  # 使用更大的长度阈值
                sentences.append(current)
                current = part
            else:
                current += part
    
    # 将剩余的片段添加到结果中
    if len(current) >= 2:
        sentences.append(current)
    
    # 如果只有一个很长的句子，尝试分成两部分
    if len(sentences) == 1 and len(sentences[0]) > min_length * 4:
        text = sentences[0]
        mid = len(text) // 2
        # 从中间向两边寻找可以分割的标点
        for i in range(mid, len(text) - 1):
            if text[i] in end_punctuations:
                sentences = [text[:i+1], text[i+1:]]
                break
    
    # 如果没有生成任何句子，返回原始文本
    if not sentences and sentence:
        sentences = [sentence]
    
    logger.debug(f"分句结果 ({len(sentences)}个): {sentences}")
    return sentences


async def gen_stream(prompt, asr = False, voice_speed=None, voice_id=None):
    logger.debug(f"生成流式响应: prompt={prompt}, voice_speed={voice_speed}, voice_id={voice_id}")
    if asr:
        chunk = {
            "prompt": prompt
        }
        yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块

    # 获取大模型回答
    text_cache = await llm_answer(prompt)
    sentences = split_sentence(text_cache)
    
    # 创建语音合成任务列表
    audio_tasks = []
    for sub_text in sentences:
        # 创建异步任务用于语音合成
        audio_tasks.append(asyncio.to_thread(get_audio, sub_text, voice_speed, voice_id))
    
    # 等待所有语音合成任务完成
    audio_results = await asyncio.gather(*audio_tasks)
    
    # 发送所有句子和对应的音频
    for index_, (sub_text, base64_string) in enumerate(zip(sentences, audio_results)):
        # 生成 JSON 格式的数据块
        chunk = {
            "text": sub_text,
            "audio": base64_string,
            "endpoint": index_ == len(sentences)-1
        }
        yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块
        await asyncio.sleep(0.05)  # 50毫秒的延迟

# 处理 ASR 和 TTS 的端点
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    # 模仿调用 ASR API 获取文本
    text = "语音已收到，这里只是模仿，真正对话需要您自己设置ASR服务。"
    # 调用 TTS 生成流式响应
    return StreamingResponse(gen_stream(text, asr=True), media_type="application/json")


async def call_asr_api(audio_data):
    # 保存成临时 WAV 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        # 使用 faster-whisper 进行识别
        segments, _info = whisper_model.transcribe(tmp_path, beam_size=5, language="zh")

        # 合并所有片段文本
        result_text = "".join([seg.text for seg in segments])
        print("识别结果：", result_text)

        return result_text

    except Exception as e:
        print("ASR识别失败：", e)
        return "识别失败"

    finally:
        os.remove(tmp_path)

        
@app.post("/eb_stream")    # 前端调用的path
async def eb_stream(request: Request):
    try:
        body = await request.json()
        input_mode = body.get("input_mode")
        voice_speed = body.get("voice_speed")
        voice_id = body.get("voice_id")

        if input_mode == "audio":
            base64_audio = body.get("audio")
            # 解码 Base64 音频数据
            audio_data = base64.b64decode(base64_audio)
            # 这里可以添加对音频数据的处理逻辑
            prompt = await call_asr_api(audio_data)  # 假设 call_asr_api 可以处理音频数据
            return StreamingResponse(gen_stream(prompt, asr=True, voice_speed=voice_speed, voice_id=voice_id), media_type="application/json")
        elif input_mode == "text":
            prompt = body.get("prompt")
            return StreamingResponse(gen_stream(prompt, asr=False, voice_speed=voice_speed, voice_id=voice_id), media_type="application/json")
        else:
            raise HTTPException(status_code=400, detail="Invalid input mode")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 添加WebSocket端点
@app.websocket("/ws_stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 等待接收消息
            data = await websocket.receive_json()
            
            # 解析接收到的数据
            input_mode = data.get("input_mode")
            voice_speed = data.get("voice_speed")
            voice_id = data.get("voice_id")
            
            logger.info(f"WebSocket接收到请求: input_mode={input_mode}, voice_speed={voice_speed}, voice_id={voice_id}")
            
            if input_mode == "audio":
                base64_audio = data.get("audio")
                # 解码 Base64 音频数据
                audio_data = base64.b64decode(base64_audio)
                logger.debug(f"WebSocket收到音频数据: {len(audio_data)} 字节")
                # 处理音频
                prompt = await call_asr_api(audio_data)

                # 获取大模型回答
                text_response = await llm_answer(prompt)
                
                # 将回答分成句子
                sentences = split_sentence(text_response)
                
                # 为每个句子单独生成语音并发送
                for index_, sub_text in enumerate(sentences):
                    # 为当前句子生成语音
                    base64_string = get_audio(sub_text, voice_speed, voice_id)
                    
                    # 发送数据
                    await manager.send_json(websocket, {
                        "text": sub_text,
                        "audio": base64_string,
                        "endpoint": index_ == len(sentences)-1
                    })
                    await asyncio.sleep(0.05)  # 减少延迟到50毫秒
                    
            elif input_mode == "text":
                prompt = data.get("prompt")
                logger.debug(f"WebSocket收到文本: {prompt}")
                
                # 获取大模型回答
                text_response = await llm_answer(prompt)
                
                # 将回答分成句子
                sentences = split_sentence(text_response)
                
                # 为每个句子单独生成语音并发送
                for index_, sub_text in enumerate(sentences):
                    # 为当前句子生成语音
                    base64_string = get_audio(sub_text, voice_speed, voice_id)
                    
                    # 发送数据
                    await manager.send_json(websocket, {
                        "text": sub_text,
                        "audio": base64_string,
                        "endpoint": index_ == len(sentences)-1
                    })
                    await asyncio.sleep(0.05)  # 减少延迟到50毫秒
            else:
                logger.warning(f"无效的输入模式: {input_mode}")
                await manager.send_json(websocket, {"error": "无效的输入模式"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.debug("WebSocket连接已断开")
    except Exception as e:
        logger.exception(f"WebSocket处理出错: {e}")
        try:
            await manager.send_json(websocket, {"error": str(e)})
        except:
            pass
        manager.disconnect(websocket)

@app.websocket("/asr_proxy")
async def asr_proxy(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 接收音频数据
            data = await websocket.receive_json()
            base64_audio = data.get("audio")
            
            if not base64_audio:
                await manager.send_json(websocket, {"error": "未收到音频数据"})
                continue
                
            # 解码 Base64 音频数据
            audio_data = base64.b64decode(base64_audio)
            logger.debug(f"ASR代理收到音频数据: {len(audio_data)} 字节")
            
            # 调用语音识别
            text = await call_asr_api(audio_data)
            
            # 返回识别结果
            await manager.send_json(websocket, {"text": text})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.debug("ASR代理WebSocket连接已断开")
    except Exception as e:
        logger.exception(f"ASR代理处理出错: {e}")
        try:
            await manager.send_json(websocket, {"error": str(e)})
        except:
            pass
        manager.disconnect(websocket)

# 启动Uvicorn服务器
if __name__ == "__main__":
    logger.info("服务器启动")
    uvicorn.run(app, host="0.0.0.0", port=8890)
