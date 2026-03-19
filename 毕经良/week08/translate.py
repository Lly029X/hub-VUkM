from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-eda26f20c01f42df8aadb6ea0d997f04",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

"""
翻译智能体 - 基于ExtractionAgent的实现
能够自动识别：
1. 原始语种
2. 目标语种
3. 待翻译的文本
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            print("提取的参数：************")
            print(arguments)

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class TranslationTask(BaseModel):
    """翻译任务信息提取"""
    source_language: str = Field(description="原始文本的语言，如中文、英文、日文等")
    target_language: str = Field(description="目标语言，如中文、英文、日文等")
    text_to_translate: str = Field(description="需要翻译的文本内容")


def translate_agent():
    """
    翻译智能体主函数
    使用ExtractionAgent自动识别翻译任务
    """
    # 创建智能体实例
    agent = ExtractionAgent(model_name="qwen-plus")

    # 测试用例1：中文翻译成英文
    print("=" * 60)
    print("测试用例1：中文翻译成英文")
    print("=" * 60)
    result1 = agent.call("把'你好，世界'翻译成英文", TranslationTask)
    print("翻译任务信息：")
    print(f"  原始语种: {result1.source_language}")
    print(f"  目标语种: {result1.target_language}")
    print(f"  待翻译文本: {result1.text_to_translate}")
    print()

    # 测试用例2：英文翻译成中文
    print("=" * 60)
    print("测试用例2：英文翻译成中文")
    print("=" * 60)
    result2 = agent.call("Please translate 'Hello, AI world' to Chinese", TranslationTask)
    print("翻译任务信息：")
    print(f"  原始语种: {result2.source_language}")
    print(f"  目标语种: {result2.target_language}")
    print(f"  待翻译文本: {result2.text_to_translate}")
    print()

    return result1, result2


if __name__ == "__main__":
    print("翻译智能体启动...")
    print("=" * 60)
    print()

    translate_agent()

    print("=" * 60)
    print("翻译智能体执行完成！")
