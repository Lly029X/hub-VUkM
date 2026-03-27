from pydantic import BaseModel, Field
import openai
import os
import json

# 初始化客户端
client = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-xxxxxxxx"),
    base_url="xxxxxxx",
)


class TranslationTask(BaseModel):
    """翻译任务"""
    source_text: str = Field(description="需要翻译的原始文本")
    source_language: str = Field(description="原始语种，如：英语、中文、日语等")
    target_language: str = Field(description="目标语种，如：中文、英语、法语等")
    translated_text: str = Field(description="翻译后的文本")


class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def translate(self, user_input: str) -> TranslationTask:
        """
        一步完成翻译意图识别和执行
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个翻译助手。分析用户输入，识别出需要翻译的文本、原始语种和目标语种，然后执行翻译。"
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "translate_text",
                    "description": "翻译文本",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_text": {
                                "type": "string",
                                "description": "原始文本"
                            },
                            "source_language": {
                                "type": "string",
                                "description": "原始语种"
                            },
                            "target_language": {
                                "type": "string",
                                "description": "目标语种"
                            },
                            "translated_text": {
                                "type": "string",
                                "description": "翻译结果"
                            }
                        },
                        "required": ["source_text", "source_language", "target_language", "translated_text"]
                    }
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                result = json.loads(arguments)
                return TranslationTask(**result)
            else:
                print("未检测到翻译意图")
                return None

        except Exception as e:
            print(f"翻译过程出错：{e}")
            return None


# 测试
if __name__ == "__main__":
    agent = TranslationAgent()

    # 测试用例
    test_input = "帮我将good！翻译为中文"
    print(f"用户输入：{test_input}")

    result = agent.translate(test_input)
    if result:
        print(f"\n识别结果：")
        print(f"原始文本：{result.source_text}")
        print(f"原始语种：{result.source_language}")
        print(f"目标语种：{result.target_language}")
        print(f"翻译结果：{result.translated_text}")