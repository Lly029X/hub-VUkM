from openai import OpenAI
import os
import fitz  # PyMuPDF 用于PDF转图片
import base64

# ---------------------- 配置 ----------------------
# 初始化OpenAI客户端（百炼兼容模式）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 本地PDF路径
PDF_FILE_PATH = "/Users/jlbi/Desktop/week10/Week09-Dify智能体搭建.pdf"

# ---------------------- 工具函数：PDF第一页转base64图片 ----------------------
def pdf_first_page_to_base64(pdf_path):
    """
    读取本地PDF，提取第一页，转成base64编码的图片
    """
    doc = fitz.open(pdf_path)
    page = doc[0]  # 取第一页
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2倍清晰度
    img_bytes = pix.tobytes("png")
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    doc.close()
    return f"data:image/png;base64,{base64_str}"

# ---------------------- 主逻辑 ----------------------
# 转换PDF第一页为base64
pdf_image_base64 = pdf_first_page_to_base64(PDF_FILE_PATH)

reasoning_content = ""  # 思考过程
answer_content = ""     # 完整回复
is_answering = False    # 是否开始输出答案

# 请求百炼视觉模型
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pdf_image_base64}  # 直接传本地PDF图片
                },
                {"type": "text", "text": "请详细解析这一页PDF的内容，包括文字、题目、图表、结构等"}
            ]
        }
    ],
    stream=True,
)

# ---------------------- 流式输出（完全对齐你原来的格式） ----------------------
print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        
        # 输出思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        
        # 输出答案
        else:
            if delta.content and not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            if delta.content:
                print(delta.content, end='', flush=True)
                answer_content += delta.content