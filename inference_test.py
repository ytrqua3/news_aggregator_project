import torch
import os
import json

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.Linear(10, 2).to(device)
    model_state_dict = torch.load(os.path.join(model_dir, 'smoke_test.bin'), map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, device

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)['input']
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model_device):
    model, device = model_device
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device) #unsqueeze (10,) -> (1, 10)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        label = torch.argmax(probabilities, axis=1).item() #item() only works on single element tensors. use tolist() otherwise
    

    return {'label': int(label), 'probabilities': probabilities.cpu().numpy().tolist()}

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"unsupported accept type: {accept}")