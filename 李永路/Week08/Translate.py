from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing import Literal
import openai

client = openai.OpenAI(
    api_key="",  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class TranslationTask(BaseModel):
    """翻译任务信息提取"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text_to_translate: str = Field(description="待翻译的文本")


class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
    
    def extract_translation_info(self, user_prompt: str):
        """
        从用户输入中提取翻译任务信息
        
        Args:
            user_prompt: 用户输入的翻译请求
            
        Returns:
            TranslationTask: 包含原始语种、目标语种和待翻译文本的对象
        """
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # 构建 tool 描述
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TranslationTask.model_json_schema()['title'],
                    "description": TranslationTask.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": TranslationTask.model_json_schema()['properties'],
                        "required": TranslationTask.model_json_schema()['required'],
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
            # 提取的参数（json 格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            
            # 参数转换为 datamodel，获取想要的参数
            return TranslationTask.model_validate_json(arguments)
        except Exception as e:
            print('ERROR:', e)
            print('Response:', response.choices[0].message)
            return None
    
    def translate(self, translation_task: TranslationTask) -> str:
        """
        执行翻译任务
        
        Args:
            translation_task: 翻译任务信息
            
        Returns:
            str: 翻译结果
        """
        messages = [
            {
                "role": "system",
                "content": f"你是一个专业的翻译助手。请将文本从{translation_task.source_language}翻译为{translation_task.target_language}。只输出翻译结果，不要有其他解释。"
            },
            {
                "role": "user",
                "content": translation_task.text_to_translate
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        
        return response.choices[0].message.content.strip()
    
    def process(self, user_prompt: str) -> dict:
        """
        处理完整的翻译流程
        
        Args:
            user_prompt: 用户输入的翻译请求
            
        Returns:
            dict: 包含翻译任务信息和翻译结果的字典
        """
        # 提取翻译任务信息
        translation_task = self.extract_translation_info(user_prompt)
        
        if not translation_task:
            return {
                "success": False,
                "error": "无法解析翻译任务信息"
            }
        
        # 执行翻译
        translation_result = self.translate(translation_task)
        
        return {
            "success": True,
            "task": translation_task,
            "result": translation_result
        }


# 使用示例
if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")
    
    # 测试用例：帮我将 good！翻译为中文
    prompt = "帮我将 good！翻译为中文"
    print(f"用户输入：{prompt}")
    print("=" * 60)
    
    result = agent.process(prompt)
    
    if result["success"]:
        task = result["task"]
        print(f"原始语种：{task.source_language}")
        print(f"目标语种：{task.target_language}")
        print(f"待翻译文本：{task.text_to_translate}")
        print(f"翻译结果：{result['result']}")
    else:
        print(f"错误：{result['error']}")
    
    print("\n" + "=" * 60)
    
    # 更多测试用例
    test_cases = [
        "Please translate '你好世界' to English",
        "把这句话翻译成日文：今天天气真好",
        "我需要将'Bonjour'从法文翻译成中文",
    ]
    
    for prompt in test_cases:
        print(f"\n用户输入：{prompt}")
        print("-" * 60)
        
        result = agent.process(prompt)
        
        if result["success"]:
            task = result["task"]
            print(f"原始语种：{task.source_language}")
            print(f"目标语种：{task.target_language}")
            print(f"待翻译文本：{task.text_to_translate}")
            print(f"翻译结果：{result['result']}")
        else:
            print(f"错误：{result['error']}")
