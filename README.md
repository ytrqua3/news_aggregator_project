This project is built to familiarize myself with trainig and deploying the classification model for the music preference project
link to project: 
So I started the extension of the term project in early January and found myself stuck in writing code to train a model in sagemaker as it is not covered in the course.
Therefore, after I finished with the simple logistic regression model that could be trained using AWS Glue, I decided to take this sagemaker course
link to course:
There is a part about load testing and creating an api endpoint for the model, but I already learnt about lambda in my previous course. So I will come back once I have finished with my music preference project.

What this project does: train and deploy a model that classifies news titles into 4 categores: business, entertainment, health, and science
  - structure of the model: feed the title into distilbert -> get CLS token (treat it as pooler) -> train a nn to classify
  - news_aggregator_training.ipynb: training job using sagemaker
  - news_aggregator_deployment.ipynb: deploy an endpoint for the model
  - script.py: code for training the fine tuned model
  - inference.py: code that defines functions to help handle inputs
  - news_aggregator_test_training.ipynb, news_aggregator_test_deployment.ipynb: notebook for train and deploy smoke test code to ensure that the training job could run
  - test.py, inference_test.py: code used to test the environment and help me have a better understanding on how aws uses instances to run code


This is a log of notes and learning points
22Jan: Started "AI Engineering: Build, Train and Deploy Models with AWS SageMaker" by Patrik Szepesi in zerotomastery.io 
  - Puropose: to learn how to deply a machine learning model using hugging face in sagemaker for my music preference project
  - deployed a trained sentiment analysis model to get an idea of how huggingface and sagemaker work together (code is not included in this repo)

24Jan: Understanding mechanisms behind neural networks
  - how neural network combines different functions to fit a curve to datapoints(Statquest)
      -> backpropagation, gradient decent, activation functions

25-27Jan: Understanding LLM (DistilBert)
  - positional encoding: by adding token embeddings to positional embeddings, the model catches pattern with context of where the token is positioned in the sentence
      -> postional embeddings are produced with sin and cos functions (parameters: pos(absolute position), i(dimension index), d(#dimentsions), n(scaling parameter))
  - attention blocks: take a sequence in and output a context aware vector for each token
      -> transform every token into query, key, and value vectors (linear transformation that is trained with certain purposes)
      -> to get the context aware vector for a token:
          1. attention mask = (query vector for the word) dot (a matrix with key vector for all tokens in sequence) => (seq_len, ) => softmax => normalization
          2. output = sum of (attention mask * value vector) for every token in the sequence => (768, )

28Jan: writing code for training notebook and script.py
  - fine tuned distilbert model:
      1. tokenize input sentence using distilbert tokenizer
      2. form embeddings of size 768 for each token
      3. pass the sequence into layers of transformers
      4. extract the output vector CLS token for sentence level representation (not trained explicitly but still yields good performance)
      5. pass through a linear layer 768->768
      6. apply ReLU to each neuron to give non linearity
      7. apply dropout (each neuron has 30% chance of turning to 0 and others are scaled up to avoid overfitting)
      8. pass the vector throguh another linear layer 768->4 where the outputs are 4 logits for the categories
  - create a custom class for the model structure (DistilBERTClass)
      -> inherits torch.nn.Module
      -> forward function defines the structure of the model
  - create a custom class for storing tokenized data (NewsDataset)
      -> allows creation of DataLoader which allowes batching and shuffling
      -> turns a row to df into a dictionary
      -> __getitem__ returns a dictionary of ids(tokenized inputs, tensor(max_len, )), mask(attention mask, tensor(512, )), and targets(category, single-element tensor)
  - create train and valid function
      -> for each patch of sequences: forward feed->calculate metrics->backpropagation->update parameters based on gradient
      -> loss function: cross entropy (how far off the prediction is from the target)
      -> model.eval(): evaluation mode ignores dropout layer
  - main funtion (flow of the code)
      1. parse arguments
      2. load the csv into pandas
      3. encode categories
      4. train test split
      5. create data loaders
      6. train the model my looping through number of epochs
      7. save the model and label encoder
  - for fine tuning models, epochs are low
  - final: completed the script but not yet debugged

29Jan: writing code for deployment notebook and inference.py
  - save label encoder in script.py as a part of the model using joblib
  - according to huggingface documentation:
  - model_fn(model_dir) overrides the default method for loading a model. The return value model will be used in predict for predictions. predict receives argument the model_dir, the path to your unzipped      model.tar.gz.
  - input_fn(input_data, content_type) overrides the default method for preprocessing. The return value data will be used in predict for predictions. The inputs are:
input_data is the raw body of your request. content_type is the content type from the request header.
  - predict_fn(processed_data, model) overrides the default method for predictions. The return value predictions will be used in postprocess. The input is processed_data, the result from preprocess.
  - output_fn(prediction, accept) overrides the default method for postprocessing. The return value result will be the response of your request (e.g.JSON). The inputs are:
predictions is the result from predict. accept is the return accept type from the HTTP Request, e.g. application/json.
  - input_fn -> model_fn -> predict_fn -> output_fn
  - final: completed the script but not yet debugged

30Jan: writing smoke test code. debugging and training the models
  - model and input tensors must be on the same device -> e.g. ids.to(device)
  - item() only works on single element tensors. use tolist() otherwise
  - DistilBertTokenizer can return tokens in tensors for different frameworks, in my case return_tensor=pt
  - I can save configurations of the model instead of reconstructing the model in inference.py
  - label encoding before train test split can cause data leakage
  - used 5% of the data to test the whole pipeline before using all the data
  - model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    saving the model structure and the tokenizer can avoid model_fn to recreate the model from scratch
  - final: completed training and deploying the model with ~95% validation accuracy in last epoch
