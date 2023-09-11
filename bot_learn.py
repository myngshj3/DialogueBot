import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

# データの前処理
tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
input_text = ["Input sentence 1", "Input sentence 2", ...]  # 学習データのリスト
target_text = ["Target sentence 1", "Target sentence 2", ...]  # ターゲットデータのリスト

# データをトークン化し、モデルの入力形式に変換
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
target_ids = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)

# モデルのセットアップ
model = T5ForConditionalGeneration.from_pretrained("t5-small")
optimizer = AdamW(model.parameters(), lr=1e-4)

# 学習ループの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

# 学習
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(input_ids=input_ids["input_ids"], labels=target_ids["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1} Loss: {loss.item()}")

# 学習したモデルを保存
model.save_pretrained("trained_t5_model")

# 学習したモデルを評価やデプロイに使用
