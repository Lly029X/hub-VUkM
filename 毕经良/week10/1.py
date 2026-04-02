import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ===================== 1. 加载本地 Chinese-CLIP 模型 =====================
MODEL_LOCAL_PATH = "/Users/jlbi/ai_study/models/AI-ModelScope/chinese-clip-vit-base-patch16"

# 本地加载模型 + 处理器
model = ChineseCLIPModel.from_pretrained(MODEL_LOCAL_PATH)
processor = ChineseCLIPProcessor.from_pretrained(MODEL_LOCAL_PATH)

# ===================== 2. 本地图片路径 =====================
image_path = "/Users/jlbi/Desktop/week10/dog.png"  # 你的小狗图片路径
image = Image.open(image_path).convert("RGB")  # 统一转RGB，避免报错

# ===================== 3. 零样本分类标签（纯中文，最适合这个模型） =====================
# 你可以随便加、随便改类别
candidate_labels = [
    "小狗",
    "小猫",
    "汽车",
    "树木",
    "花朵",
    "小鸟",
    "人类",
    "水果",
    "桌子",
    "椅子"
]

# ===================== 4. 预处理输入 =====================
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# ===================== 5. 模型推理 =====================
with torch.no_grad():
    outputs = model(**inputs)

# 计算相似度 -> 概率
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# ===================== 6. 输出结果 =====================
print("=" * 40)
print("         Chinese-CLIP 零样本图像分类结果")
print("=" * 40)

# 打印所有类别概率
for label, prob in zip(candidate_labels, probs[0]):
    print(f"【{label}】: {prob:.4f}")

# 输出最优结果
best_idx = probs.argmax().item()
best_label = candidate_labels[best_idx]
best_prob = probs[0][best_idx].item()

print("\n" + "=" * 40)
print(f"🏆 最终识别结果：{best_label}")
print(f"📊 置信度：{best_prob:.2%}")
print("=" * 40)