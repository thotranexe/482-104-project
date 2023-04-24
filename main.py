import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import re
from tqdm.notebook import tqdm
from typing import *
import string
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AdamW
from transformers import DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification
import streamlit as st
st.write("Please be patient model training takes 20+ mins :P")
#config constants
SEED = 42
EPOCHS = 2
SEQ_SIZE = 150
BATCH_SIZE = 32
PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"

#import all data
data=pd.read_csv('./data/train.csv',engine='python',encoding='utf-8', error_bad_lines=False)
test=pd.read_csv('./data/test.csv',engine='python',encoding='utf-8', error_bad_lines=False)
test_labels=pd.read_csv('./data/test_labels.csv',engine='python',encoding='utf-8', error_bad_lines=False)
sub=pd.read_csv('./data/sample_submission.csv',engine='python',encoding='utf-8', error_bad_lines=False)

#setup data
data.drop(columns='id',inplace=True)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#text proccessing
def cleanString(comment: str) -> str:
    #contrationcs
    comment = re.sub('n\'t', ' not', comment) 
    comment = re.sub('\'m', ' am', comment)
    comment = re.sub('\'ve', ' have', comment)
    comment = re.sub('\'s', ' is', comment)
    #newline
    comment = comment.replace('\n', ' \n ')
    comment = comment.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    comment = comment.replace(r'[0-9]', '') 
    comment = re.sub('[^a-zA-Z%]', ' ', comment)
    comment = re.sub('%', '', comment)
    comment = re.sub(r' +', ' ', comment)
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r' +', ' ', comment)
    comment = comment.strip()
    return comment

data.comment_text=data.comment_text.map(cleanString)

#tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []

for txt in tqdm(data.comment_text):
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

#test train split
df_train, df_test = train_test_split(data, test_size=0.15, random_state=SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=SEED)
#set pytorch dataset
class CommentDataset(Dataset):
    def __init__(self, comments, targets, tokenizer, max_len):
        assert len(comments) == len(targets)
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(comment,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                            #   padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                             )
        return {'review_text': comment,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)}

def create_data_loader(df: pd.DataFrame, tokenizer, max_len: int, batch_size: int):
    ds = CommentDataset(comments=df.comment_text.to_numpy(),
                        targets=df[labels].to_numpy(),
                        tokenizer=tokenizer,
                        max_len=max_len)

    return DataLoader(ds, batch_size=batch_size)

#helper function to set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(SEED)

#gpu usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = DistilBertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
config.num_labels = len(labels)
config.problem_type = "multi_label_classification"
config.classifier_dropout = 0.2
config.return_dict = True

model = DistilBertForSequenceClassification(config)
model = model.to(device)

train_dataloader = create_data_loader(df=df_train, tokenizer=tokenizer, max_len=SEQ_SIZE, batch_size=BATCH_SIZE)
val_dataloader = create_data_loader(df=df_val, tokenizer=tokenizer, max_len=SEQ_SIZE, batch_size=1)
test_dataloader = create_data_loader(df=df_test, tokenizer=tokenizer, max_len=SEQ_SIZE, batch_size=1)

def train_epoch_for_hf(model, data_loader: DataLoader, device: torch.device, optimizer):
    """
    hf = huggingface.
    """
    model.train()

    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].float().to(device)
        
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def evaluate_for_hf(model, data_loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    score = None

    for idx, batch in enumerate(tqdm(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].float().to(device)
        with torch.set_grad_enabled(False):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            if idx == 0:
                score =  outputs.logits.cpu()
            else:
                score = torch.cat((score, outputs.logits.cpu()))
            losses.append(outputs.loss.item())
    return score, np.mean(losses)

optimizer = AdamW(model.parameters(), lr=2e-5)
best_val_loss = 9999.
print('====START TRAINING====')
#training here
for epoch in tqdm(range(EPOCHS)):
     print('-' * 10)
     train_epoch_for_hf(model=model, data_loader=train_dataloader, optimizer=optimizer, device=device)
     _, tr_loss = evaluate_for_hf(model=model, data_loader=train_dataloader, device=device)
     val_pred, val_loss = evaluate_for_hf(model=model, data_loader=val_dataloader, device=device)
     y_pred_np = val_pred.numpy()
     val_auc = roc_auc_score(df_val[labels].to_numpy(), y_pred_np)
     if val_loss < best_val_loss:
         best_val_loss = val_loss
         #torch.save(model.state_dict(), 'distill_bert.pt')
     print(f'Epoch {epoch + 1}/{EPOCHS}', f'train loss: {tr_loss:.4},', f'val loss: {val_loss:.4},', f'val auc: {val_auc:.4}')
# once model is saved and generated no need to re run :)
#model = DistilBertForSequenceClassification(config)
#model.load_state_dict(torch.load('./distill_bert.pt'))
#model = model.to(device)
#test model here
test_pred, test_loss = evaluate_for_hf(model=model, data_loader=test_dataloader, device=device)
print('====TEST RESULT====')
print(f'Log loss: {test_loss:.5}')
y_pred_np = test_pred.numpy()
test_auc = roc_auc_score(df_test[labels].to_numpy(), y_pred_np)
print(f'ROC AUC: {test_auc:.5}')

test_src_id = test.iloc[:, 0]
test.drop(columns='id', inplace=True)
test_labels.drop(columns='id', inplace=True)
test_src = pd.concat((test, test_labels), axis=1)

test_src_dataloader = create_data_loader(df=test_src, tokenizer=tokenizer, max_len=SEQ_SIZE, batch_size=1)
prediction, _ = evaluate_for_hf(model=model, data_loader=test_src_dataloader, device=device)
prediction = torch.sigmoid(prediction).numpy()

sub[labels] = prediction
sub.insert(1,"tweet",data.comment_text,True)
st.daatframe(sub)