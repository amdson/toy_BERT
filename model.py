from torch import Tensor
import torch
import torch.nn as nn
import math

from vocab import EXPR_TOKENIZER, PAD_IDX
from data_gen import create_masks

#TODO Make sure switch to batch first is handled correctly
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        print("pos embedding", pos_embedding.shape, pos_embedding)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# BERT Network
from torch.nn import TransformerEncoder
class BERTEncoder(nn.Module):
    def __init__(self, num_encoder_layers: int, emb_size: int, nhead: int, vocab_size: int, 
                    dim_feedforward: int = 512, dropout: float = 0.1):
        super(BERTEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.generator = nn.Linear(emb_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.tok_emb(src))
        outs = self.transformer_encoder(src_emb, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        src_emb = self.positional_encoding(self.tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask)

import pytorch_lightning as pl
class BERTModule(pl.LightningModule):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.model = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, nhead, 
                                            src_vocab_size, tgt_vocab_size, dim_feedforward, dropout)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(self, src: Tensor, trg: Tensor,
                src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor):
        return self.model(src, trg, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask)

    def training_step(self, batch, batch_nb):
        src, src_pad_mask, tgt, tgt_pad_mask = batch
        tgt_input = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        src_mask, tgt_mask = create_masks(src, tgt_input, src.get_device())
        logits = self.model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
        tgt_label = tgt[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_label.reshape(-1))
        self.log("loss", loss, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_nb):
        src, src_pad_mask, tgt, tgt_pad_mask = batch
        # print(src.shape, src_pad_mask.shape, tgt.shape, tgt_pad_mask.shape)

        tgt_input = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        src_mask, tgt_mask = create_masks(src, tgt_input, src.get_device())
        # print("tgt_mask", tgt_mask, tgt_mask.shape)
        # print("src_mask", src_mask, src_mask.shape)
        # print("src_key_padding_mask", src_pad_mask)
        # print("tgt_key_padding_mask", tgt_pad_mask)
        logits = self.model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
        tgt_label = tgt[:, 1:]
        # print(logits.shape, tgt.shape)
        # if(batch_nb < 5):
        #     print("validation logits", logits, "tgt", tgt)
        if(batch_nb == 0):
            src_l = src.tolist()
            tgt_preds = torch.argmax(logits, 2).tolist()
            src_d = EXPR_TOKENIZER.decode_batch(src_l)
            tgt_d = EXPR_TOKENIZER.decode_batch(tgt_preds)
            for i, s, t in zip(range(5), src_d, tgt_d):
                print(f"{s} = {t}")
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_label.reshape(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True) 
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)

