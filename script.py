import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#custom Dataset class
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer,
        self.len = len(df),
        self.max_len = max_len,

    def __getitem__(self, index):
        title = " ".join(str(self.data.iloc[0, index]).split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )

        ids = inputs['inpput_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[2, index], dtype=torch.long)
        }

    def __len__(self):
        return self.len


#create class for the model
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__
        self.l1=DistilBertModel.from_pretrained('distilbert-base-uncased') #base model
        self.pre_classifier = torch.nn.Linear(768,768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768,4) #outputs 4 possible categories

    #defines the structure of the neural network
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask) #must use self.l1 so the model doesnt "forget"
        hidden_state = output_1.last_hidden_state #contains one contextual embedding for each token (batch_size, seq_len, 768 dim embedding)
        pooler = hidden_state[:, 0, :] #(batch_size, 768), uses CLS token as pooler (trained to represent the whole sentence)
        #BERT forces [CLS] to be sentence-level via a pooler and training objectives; DistilBERT allows [CLS] to summarize the sentence, but does not explicitly train it to do so.
        #feed pooler into the self defined neural network
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def train(epoch, model, device, training_loader, optimizer, loss_function):
    true_loss = 0
    n_correct = 0
    true_steps = 0
    n_examples = 0

    model.train() #It does not train the model by itself. It just switches the model into training mode.

    #loops through batches of sequences in the training_loader
    for _, data in enumerate(training_loader, 0):
        #data will be dictionary returned in __getitem__ from NewsDataset
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        #forward pass
        output = model(ids, mask) #trained to be in the exact order defined by your dataset’s label encoding

        #calculate metrics
        true_loss += loss_function(output, targets).item()
        true_steps += 1
        big_val, big_idx = torch.max(output.data, dim=1) #(batch_size,)
        n_correct += (big_idx == targets).sum().items()
        n_examples += targets.size(0) #increment by batch size

        #every 5000 batches
        if _%5000 == 0:
            print(f"Training loss: {true_loss/true_steps}")
            print(f"Training Accuracy: {n_correct/n_examples}")

        #backward
        optimizer.zero_grad() #reset gradient
        loss.backward() #backward propagation
        optimizer.step() #make changes to parameters based on gradient

    print(f"Epoch {epoch} Training loss: {true_loss/true_steps}")
    print(f"Epoch {epoch} Training Accuracy: {n_correct/n_examples}")

def valid(epoch, model, testing_loader, device, loss_function):
    true_loss = 0
    n_correct = 0
    true_steps = 0
    n_examples = 0

    model.eval()

    with torch.no_grad(): #disables gradient computation (not needed for validation)
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask)

            true_loss += loss_function(output, targets).item()
            true_steps += 1
            big_val, big_idx = torch.max(output.data, dim=1)
            n_correct += (big_idx == targets).sum().items()
            n_examples += targets.size(0)

            if _%1000 == 0:
                print(f"Validation loss: {true_loss/true_steps}")
                print(f"Validation Accuracy: {n_correct/n_examples}")

    print(f"Epoch {epoch} Validation loss: {true_loss/true_steps}")
    print(f"Epoch {epoch} Validation Accuracy: {n_correct/n_examples}")

def main():
    print("start script.py")

    #fetch arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=int, default=5e-5)
    parser.add_argument("--data_path", type=str, default="news-aggregator-sadrian-bucket/newsCorpora.csv")
    parser.add_argument("--seq_max_len", type=int, default=512)

    args = parser.parse_args()

    #get dataframe
    df = pd.read_csv(f's3://{args.data_path}', sep='\t', names=['ID', 'TITLE', 'URL', 
    'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    df = df[['TITLE', 'CATEGORY']]

    #encode different categories
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['CATEGORY'])
    print("Successfully labelled data")

    #train test split
    train_size = 0.8
    train_df = df.sample(frac=train_size, random_state=0)
    valid_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    print("train test split performed")
    print(f"Full dataset:{df.shape}")
    print(f"train dataset:{train_df.shape}")
    print(f"validaiton dataset:{valid_df.shape}")

    #create loaders
    train_dataset = NewsDataset(train_df, tokenizer, args.seq_max_len)
    valid_dataset = NewsDataset(valid_df, tokenizer, args.seq_max_len)

    train_parameters={
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    valid_parameters={
        'batch_size': args.valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_dataset, **train_parameters)
    valid_loader = DataLoader(valid_dataset, **valid_parameters)
    print("data loaders created successfully")

    #define elements for training
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertClass()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params = model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropy()

    #train loop
    print("Starting training process")
    for epoch in range(EPOCHS):
        print(f"starting Epoch {epoch}...")
        train(epoch, model, device, training_loader, optimizer, loss_function)
        valid(epoch, model, device, testing_loader, loss_function)
    print("model successfully trained")

    #save the model
    output_dir = os.environ['SM_MODEL_DIR']

    print("saving model into s3")
    output_model_path = os.join(output_dir, 'pytorch_distilbert_news.bin')
    torch.save(model.state_dict(), output_model_path)
    output_vocab_path = os.join(output_dir, 'vocab_distilbert_news.bin')
    tokenizer.save_vocabulary(output_vocab_file)
    print("sucessfully ended script.py")

if __name__ == '__main__':
    main()

