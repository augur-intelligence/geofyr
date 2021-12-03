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
from sklearn.model_selection import train_test_split
from pathlib import Path
LOGDIR = Path("logs")
LOGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOGDIR.joinpath("train.log"),  level=logging.DEBUG)

## MODEL PARAMS
BASE_MODEL = 'distilbert-base-uncased'
TOKEN_MODEL = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 200
NUM_LABELS = 2
TRAIN_BATCH_SIZE = 120
TEST_BATCH_SIZE = 120
NEPOCHS = 40
TEXTBATCHES = 2000
INFO = 'wiki_exploded_geonames'
DATE = str(dt.now().date())
LOGSTR = f"{DATE}_model-{TOKEN_MODEL}_loss-{INFO}"
CHECKPOINT = TOKEN_MODEL
CHECKPOINT_DIR = Path(f"checkpoints/{LOGSTR}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

## PREP DATA LOADERS
df = pd.read_parquet("data/geo_data.parquet")
texts = df["text"].values.tolist()
labels = df[["lat",  "lon"]].astype(float).values.tolist()
train_ratio = 0.78
test_ratio = 0.17
validation_ratio = 0.5
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=1 - train_ratio)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKEN_MODEL)
train_dataset = StreamTokenizedDataset(x_train, y_train, tokenizer, TEXTBATCHES, MAX_SEQ_LENGTH)
test_dataset = StreamTokenizedDataset(x_test, y_test, tokenizer, TEXTBATCHES, MAX_SEQ_LENGTH)
val_dataset = StreamTokenizedDataset(x_val, y_val, tokenizer, TEXTBATCHES, MAX_SEQ_LENGTH)

## INIT MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT)
model.config.max_position_embeddings = MAX_SEQ_LENGTH
model.num_labels = NUM_LABELS
# model = nn.DataParallel(model)
model.to(device)
torch.save(model, CHECKPOINT_DIR.joinpath("model.pt"))
GS_PATH = "gs://geobert/" + str(CHECKPOINT_DIR.joinpath("model.pt"))
fs.upload(str(CHECKPOINT_DIR.joinpath("model.pt")), str(GS_PATH))

## INIT HELPERS
optim = AdamW(model.parameters(), lr=5e-5)
writer = SummaryWriter(log_dir="gs://geobert/logs")
early_stopping = EarlyStopping(
    patience=5,
    verbose=True,
    path=CHECKPOINT_DIR.joinpath("model.pt"),
    trace_func=logging.info)

## START TRAINING
for epoch in range(0, NEPOCHS):
    ###############
    # Start epoch #
    ###############
    logging.info(f"Starting epoch {epoch}.")
    train_losses = []
    val_losses = []
    ##################
    # Train in epoch #
    ##################
    ### Train data process
    logging.info(f"Load training data.")
    train_loader = iter(DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=1))
    logging.info(f"Starting training.")
    model.train()
    for iteration, batch in enumerate(train_loader):
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
        ### Logging
        train_loss_float = float(train_loss) 
        train_losses.append(train_loss_float)
        writer.add_scalar(LOGSTR + "-train", train_loss_float, iteration)
        logging.info(f"E:{epoch:3d}, I:{iteration:8d} TRAIN: {train_loss_float:10.3f}")
        del input_ids, attention_mask, labels, logits, train_loss

    #################
    # Eval in epoch #
    #################
    ### Test data process
    test_loader = iter(DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=1))
    logging.info(f"Starting evaluation.")
    model.eval()
    with torch.no_grad():
        for iteration, val_batch in enumerate(test_loader):
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
            ### Logging
            val_loss_float = float(val_loss)
            val_losses.append(val_loss_float)
            writer.add_scalar(LOGSTR + "-test", val_loss_float, iteration)
            logging.info(f"E:{epoch:3d}, I:{iteration:8d} TEST: {val_loss_float:10.3f}")
            del val_input_ids, val_attention_mask, val_labels, val_logits, val_loss

    ################
    # Finish epoch #
    ################
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