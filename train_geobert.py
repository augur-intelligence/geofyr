import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, AdamW
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from datetime import datetime as dt
from utils.utils import *
import webdataset as wds
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(filename='train.log',  level=logging.DEBUG)

## MODEL
BASE_MODEL = 'distilbert-base-uncased'
TOKEN_MODEL = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 500
NUM_LABELS = 2
TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NEPOCHS = 40
LOSS = 'huber'
DATE = str(dt.now().date())
LOGSTR = f"{DATE}_model-{TOKEN_MODEL}_loss-{LOSS}"
CHECKPOINT = TOKEN_MODEL
CHECKPOINT_DIR = f"checkpoints/{LOGSTR}"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = DistilBertConfig()
config.num_labels = NUM_LABELS
config.max_position_embeddings = MAX_SEQ_LENGTH

model = DistilBertForSequenceClassification(config).from_pretrained(CHECKPOINT)
# model = nn.DataParallel(model)
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)
writer = SummaryWriter(log_dir="gs://geobert/logs")
early_stopping = EarlyStopping(patience=5, verbose=True, path=CHECKPOINT_DIR, trace_func=logging.info)

for epoch in range(0,NEPOCHS):
    logging.info(f"Starting epoch {epoch}.")
    train_losses = []
    val_losses = []
    # Train in epoch
    # Train data process
    logging.info(f"Load training data.")
    signed_train_url = generate_signed_url(CRED_PATH, 'geobert', 'data/train_geo_wds.tar')
    train_dataset = (wds
                     .WebDataset(signed_train_url)
                     .decode('torch'))
    train_loader = iter(DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=1))
    logging.info(f"Starting training.")
    model.train()
    for iteration, files in enumerate(train_loader):
        try:
            optim.zero_grad()
            batch = files['enc_dict.pyd']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            logits = outputs.get('logits')
            loss_fct = nn.HuberLoss()
            train_loss = loss_fct(logits, labels)
            train_loss.backward()
            optim.step()
            # Logging
            train_loss_float = float(train_loss) 
            train_losses.append(train_loss_float)
            writer.add_scalar(LOGSTR + "-train", train_loss_float, iteration)
            logging.info(f"E:{epoch:3d}, I:{iteration:8d} TRAIN: {train_loss_float:10.3f}")
            del input_ids, attention_mask, labels, logits, train_loss
        except Exception as e:
            logging.exception(e)

        
    # Eval in epoch
    # Test data process
    signed_test_url = generate_signed_url(CRED_PATH, 'geobert', 'data/test_geo_wds.tar')
    test_dataset = (wds
                    .WebDataset(signed_test_url)
                    .decode('torch'))
    test_loader = iter(DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=1))
    logging.info(f"Starting evaluation.")
    model.eval()
    with torch.no_grad():
        for iteration, files in enumerate(test_loader):
            try:
                val_batch = files['enc_dict.pyd']
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
                # Logging
                val_loss_float = float(val_loss)
                val_losses.append(val_loss_float)
                writer.add_scalar(LOGSTR + "-test", val_loss_float, iteration)
                logging.info(f"E:{epoch:3d}, I:{iteration:8d} TEST: {train_loss_float:10.3f}")        
                del val_input_ids, val_attention_mask, val_labels, val_logits, val_loss
            except Exception as e:
                logging.exception(e)


    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    writer.add_scalar(LOGSTR + "-train_epoch", avg_train_loss, epoch)
    writer.add_scalar(LOGSTR + "-test_epoch", avg_val_loss, epoch)
    logging.info(f"E:{epoch:3d}, TRAIN: {avg_train_loss:10.3f}, TEST: {avg_val_loss:10.3f}")
    early_stopping(val_loss=avg_val_loss, model=model)
        
    if early_stopping.early_stop:
        logging.info("Early stopping")
        break
        
logging.info(f"Training finished in epoch {epoch}")    
model.eval()