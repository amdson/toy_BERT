from data_gen import ExpressionDataset, collate_fn, EXPR_TOKENIZER
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model import Seq2SeqModule
from pytorch_lightning.callbacks.progress import TQDMProgressBar

fds = ExpressionDataset.from_csv("data/strat_relation.csv")
l = len(fds)
a = l *4 //5
b = l//5
c = l - a - b
train_ds, val_ds, test_ds = random_split(fds, [a, b, c], generator=torch.Generator().manual_seed(1))
train_dl = DataLoader(train_ds, shuffle=True, batch_size=256, collate_fn=collate_fn)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=512, collate_fn=collate_fn)


SRC_VOCAB_SIZE = len(EXPR_TOKENIZER.get_vocab())
TGT_VOCAB_SIZE = len(EXPR_TOKENIZER.get_vocab())
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 256
NUM_ENCODER_LAYERS = 16
NUM_DECODER_LAYERS = 16

device = torch.device('cuda')
seq2seq_module = Seq2SeqModule(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
seq2seq_module = seq2seq_module.to(device)

# Initialize a trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=250,
    callbacks=[TQDMProgressBar(refresh_rate=20)],   
)

# Train the model
trainer.fit(seq2seq_module, train_dataloaders=train_dl, val_dataloaders=val_dl)