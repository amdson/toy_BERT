from data_gen import ExpressionDataset, bert_collate_strs, EXPR_TOKENIZER
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model import BERTModule
from pytorch_lightning.callbacks.progress import TQDMProgressBar

fds = ExpressionDataset.from_csv("data/equation_dataset.csv")
l = len(fds)
a = l *4 //5
b = l//5
c = l - a - b
train_ds, val_ds, test_ds = random_split(fds, [a, b, c], generator=torch.Generator().manual_seed(1))
train_dl = DataLoader(train_ds, shuffle=True, batch_size=3, collate_fn=bert_collate_strs)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=512, collate_fn=bert_collate_strs)


VOCAB_SIZE = len(EXPR_TOKENIZER.get_vocab())
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 256
NUM_ENCODER_LAYERS = 16

bert_module = BERTModule(NUM_ENCODER_LAYERS, EMB_SIZE, NHEAD, VOCAB_SIZE, dim_feedforward=FFN_HID_DIM, dropout=0)

# # Initialize a trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[1],
    max_epochs=250,
    callbacks=[TQDMProgressBar(refresh_rate=20)],   
)

# # Train the model
trainer.fit(bert_module, train_dataloaders=train_dl, val_dataloaders=val_dl)