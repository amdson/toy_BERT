from torch import Tensor
import torch
import torch.nn as nn
import math
import utils

from vocab import EXPR_TOKENIZER, PAD_IDX

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
class BERTEncoder(nn.Module):
    def __init__(self, num_encoder_layers: int, emb_size: int, nhead: int, vocab_size: int, 
                    dim_feedforward: int = 512, dropout: float = 0.1):
        super(BERTEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.tok_pred = nn.Linear(emb_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor, src_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.tok_emb(src)) #N x L x E
        outs = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask) #N x L x E
        return self.tok_pred(outs) #N x L x V

    def encode(self, src: Tensor, src_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.tok_emb(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

import pytorch_lightning as pl
class BERTModule(pl.LightningModule):
    def __init__(self, num_encoder_layers: int, emb_size: int, nhead: int, vocab_size: int, 
                    dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = BERTEncoder(num_encoder_layers, emb_size, nhead, vocab_size, 
                    dim_feedforward = dim_feedforward, dropout = dropout)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(self, src: Tensor, src_padding_mask: Tensor):
        return self.model(src, src_padding_mask)

    def training_step(self, batch, batch_nb):
        masked_src_id, masked_token_mask, src_id, src_pad_mask = batch
        logits = self.model(masked_src_id, src_pad_mask) #NxLXV

        target = src_id #NXL
        target[~masked_token_mask] = PAD_IDX #Ignore unmasked tokens when calculating loss

        loss = self.loss_fn(logits.transpose(1, 2),  target)
        self.log("loss", loss, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_nb):
        masked_src_id, masked_token_mask, src_id, src_pad_mask = batch
        logits = self.model(masked_src_id, src_pad_mask) #NxLXV

        target = src_id #NXL
        target[~masked_token_mask] = PAD_IDX #Ignore unmasked tokens when calculating loss

        loss = self.loss_fn(logits.transpose(1, 2),  target)
        accuracy = utils.token_accuracy(logits, target, masked_token_mask)
        self.log("loss", loss, on_epoch=True)
        self.log("accuracy", accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)

