import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import os
import json

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1=DistilBertModel.from_pretrained('distilbert-base-uncased') 
        self.pre_classifier = torch.nn.Linear(768,768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768,4) 

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask) 
        hidden_state = output_1.last_hidden_state 
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def model_fn(model_dir):
    print(f"Loading model from directory: {model_dir}")
    model = DistilBERTClass()
    model_state_dict = torch.load(os.path.join(model_dir, 'pytorch_distilbert_news.bin'), map_location='cpu')
    model.load_state_dict(model_state_dict)
    
    le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, le, tokenizer, device

def predict_fn(input_data, model_le_tokenizer_device):
    print("Making predictions...")
    model, le, tokenizer, device = model_le_tokenizer_device

    inputs = tokenizer(input_data, return_tensors='pt', max_len=512, padding='max_length', truncation=True).to(device) #pt: pytorch tensors as return format

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(ids, mask) #logits

    probabilities = torch.softmax(outputs, dim=1).cpu().numpy() #logits(cuda tensor) to torch tensor to numpy array

    pred = probabilities.argmax(axis=1)
    pred_label = le.inverse_transform(pred)[0]
    return {'predicted_label': pred_label, 'probabilities': probabilities.tolist()}

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"unsupported accept type: {accept}")