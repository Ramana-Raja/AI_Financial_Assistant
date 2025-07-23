from sentence_transformers import InputExample, SentenceTransformer, losses
import pandas as pd
from torch.utils.data import DataLoader

df = pd.read_csv("C:\\Users\\raman\\Downloads\\Financial-QA-10k.csv\\Financial-QA-10k.csv")

df = df.dropna(subset=["question", "context", "answer"])

answers = df["answer"].astype(str).tolist()
questions = df["question"].astype(str).tolist()
train_examples = [InputExample(texts=[q, a]) for q, a in zip(questions, answers)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    show_progress_bar=True,
)

model.save("fine-tuned-e5-finance")