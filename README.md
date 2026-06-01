# NeoVox — 轻量化数字人 AI 语伴系统

## 项目简介

基于 [kleinlee/DH_live](https://github.com/kleinlee/DH_live) 开源框架二次开发，主要改动包括：
- 重新接入 ZhipuAI（GLM-4）作为对话大模型
- 接入 Azure TTS 实现多音色语音合成
- 接入 Faster-Whisper 实现本地语音识别（ASR）
- 修复原框架若干 bug（CORS 跨域、音频格式、流式输出等）
- 新增 FastAPI 后端，支持 HTTP 流式接口与 WebSocket 双通道

## 核心功能

- 🎙️ **语音识别（ASR）**：基于 Faster-Whisper，支持中英文
- 🤖 **对话生成（LLM）**：接入智谱 AI GLM-4，支持多轮对话
- 🔊 **语音合成（TTS）**：Azure TTS，支持多音色切换
- 👤 **数字人驱动**：DH_live_mini 轻量方案，WebGL 渲染，39 Mflops/帧
- ⚡ **秒级响应**：WebSocket 流式通信，全链路异步并行

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | Python / FastAPI / Uvicorn |
| 语音识别 | faster-whisper |
| 大语言模型 | ZhipuAI GLM-4 |
| 语音合成 | Azure Cognitive Services TTS |
| 数字人渲染 | DH_live_mini / WebGL / WebCodecs |
| 前端 | HTML / JavaScript / Canvas |

## 快速开始

**1. 安装依赖**
```bash
pip install -r requirements.txt
```

**2. 安装 Azure Speech SDK**

```bash
pip install azure-cognitiveservices-speech
```

前往 [Azure 控制台](https://portal.azure.com/) 创建「语音服务」资源，获取 Key 和区域。

**3. 配置环境变量**

复制 `.env.example` 为 `.env`，填入你的 API Key：

```
ZHIPUAI_API_KEY=你的智谱AI Key
AZURE_SPEECH_KEY=你的Azure语音Key
AZURE_SPEECH_REGION=southeastasia
```

**4. 启动**
```bash
python app.py
```
访问 `http://localhost:8890/static/MiniLive_new.html`

## 致谢

原始框架：[kleinlee/DH_live](https://github.com/kleinlee/DH_live)
