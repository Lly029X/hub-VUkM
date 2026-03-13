import openai # 调用大模型
import json

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-8b47344f618342eaa3fdbab260e9e7a1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": """你是一个专业的信息解析助手。你的任务是从用户的对话文本中提取【领域(domain)】、【意图(intent)】和【槽位(slots)】信息。

### 你需要识别以下信息：
1. **领域(domain)**：识别对话所属的类别（如：app, music, telephone 等）。
2. **意图(intent)**：识别用户的具体动作（如：LAUNCH, QUERY, CALL 等）。
3. **槽位(slots)**：识别并提取对话中的关键实体信息，并以键值对的形式列出（如：{"name": "uc", "artist": "我"}）。

### 输出结果：
请严格按照以下 JSON 格式输出解析结果：
{
  "domain": "...",
  "intent": "...",
  "slots": {
    "key": "value",
    ...
  }
}"""},
        {"role": "user", "content": "雨果在自驾游的路上车子抛锚了"},
    ],
)
result = completion.choices
print(result)

"""
```json
{    
    "domain": "bus",
    "intent": "QUERY",
    "slots": {
        "startLoc_city": "许昌",
        "endLoc_city": "中山"
    }
}
```
"""