import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, AdamW
from torch import nn
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AdamW
from datetime import datetime as dt
from utils.utils import *
import webdataset as wds
from tensorboardX import SummaryWriter

## MODEL
BASE_MODEL = 'distilbert-base-uncased'
TOKEN_MODEL = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 500
NUM_LABELS = 2
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
NEPOCHS = 15
LOSS = 'huber'
DATE = str(dt.now().date())
LOGSTR = f"{DATE}_model-{TOKEN_MODEL}_loss-{LOSS}"
CHECKPOINT = '2021-11-20_model-distilbert-base-uncased_loss-huber_epoch-0'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = DistilBertConfig()
config.num_labels = NUM_LABELS
config.max_position_embeddings = MAX_SEQ_LENGTH

model = DistilBertForSequenceClassification(config).from_pretrained(CHECKPOINT)
# model = nn.DataParallel(model)
model.to(device)

signed_train_url = generate_signed_url(CRED_PATH, 'geobert', 'data/train_geo_wds.tar')
signed_test_url = generate_signed_url(CRED_PATH, 'geobert', 'data/test_geo_wds.tar')

train_dataset = (wds
                 .WebDataset(signed_train_url)
                 .repeat(nepochs=NEPOCHS)
                 .decode('torch'))
test_dataset = (wds
                .WebDataset(signed_test_url)
                .repeat()
                .decode('torch'))

train_loader = iter(DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=1))
test_loader = iter(DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=1))

optim = AdamW(model.parameters(), lr=5e-5)

writer = SummaryWriter(log_dir="gs://geobert/logs")

losses = []
iteration = 0
for epoch in list(range(1,NEPOCHS)):
    for files in train_loader:
        iteration +=1
        batch = files['enc_dict.pyd']
        model.train()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        logits = outputs.get('logits')
        loss_fct = nn.HuberLoss()
        train_loss = loss_fct(logits, labels)
        train_loss.backward()
        optim.step()
        train_loss_float = float(train_loss) 
        del input_ids, attention_mask, labels, logits, train_loss
        
        model.eval()
        with torch.no_grad():
            test_files = next(test_loader)
            val_batch = test_files['enc_dict.pyd']
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)
            val_outputs = model(
                val_input_ids, 
                attention_mask=val_attention_mask, 
                labels=val_labels, 
                output_hidden_states=True)
            val_logits = val_outputs.get('logits')
            val_loss = loss_fct(val_logits, val_labels)
            val_loss_float = float(val_loss)
            del val_input_ids, val_attention_mask, val_labels, val_logits, val_loss
        
        losses.append([
            iteration, 
            train_loss_float, 
            val_loss_float
        ])
        print(f"E:{epoch:3d}, I:{iteration:8d}TRAIN: {train_loss_float:10.3f}, VAL: {val_loss_float:10.3f}")
        writer.add_scalar(LOGSTR + "-train", train_loss_float, iteration)
        writer.add_scalar(LOGSTR + "-test", val_loss_float, iteration)
        
        
    MODELDIR = LOGSTR + f"_epoch-{epoch}"
    model.save_pretrained(MODELDIR)
    fs.upload(f'{MODELDIR}/*', f"gs://geobert/checkpoints/{MODELDIR}")
    
model.eval()