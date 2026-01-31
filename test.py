import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    print("🚀start script.py")

    #fetch arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=int, default=5e-5)
    parser.add_argument("--data_path", type=str, default="news-aggregator-sadrian-bucket/newsCorpora.csv")
    parser.add_argument("--seq_max_len", type=int, default=512)

    args = parser.parse_args()
    print(f"epochs: {args.epochs}, train_batch_size{args.train_batch_size}, valid_batch_size: {args.valid_batch_size}, learning_rate: {args.learning_rate}, data_path: {args.data_path}, seq_max_len: {args.seq_max_len}")

    print("🔍 Environment check")
    print("SM_MODEL_DIR:", os.environ.get("SM_MODEL_DIR"))
    print("SM_OUTPUT_DATA_DIR:", os.environ.get("SM_OUTPUT_DATA_DIR"))

    print("🔥 PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Linear(10, 2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    #train loop
    print("🧪Starting dummy training process")
    for epoch in range(args.epochs):
        print(f"starting Epoch {epoch}...")
        ids = torch.randn(args.train_batch_size, 10).to(device) #(batch_size, 10)
        targets = torch.randint(0, 2, (args.train_batch_size, )).to(device)
        outputs = model(ids)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss.item():.4f}")

    print("model successfully trained")

    #save the model
    output_dir = os.environ['SM_MODEL_DIR']

    print("saving model into s3")
    output_model_path = os.path.join(output_dir, 'smoke_test.bin')
    torch.save(model.state_dict(), output_model_path)
    print(f"✅ Model saved to {output_model_path}")
    print("🎉 test.py completed successfully")

if __name__ == '__main__':
    main()
