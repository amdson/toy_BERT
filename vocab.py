
from collections import OrderedDict
# import tokenizers
# from tokenizers import BertWordPieceTokenizer
# EXPR_TOKENIZER = BertWordPieceTokenizer(unk_token='[UNK]', sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]', mask_token='[MASK]')
# EXPR_TOKENIZER.enable_padding()
# special_symbols = ['[UNK]', '[SEP]', '[CLS]', '[PAD]', '[MASK]', '[BOS]', '[EOS]']
# EXPR_TOKENIZER.add_special_tokens(special_symbols)
# EXPR_TOKENIZER.add_tokens(list("0123456789+-*/()xyz"))
# 
# EXPR_TOKENIZER.post_processor = tokenizers.processors.BertProcessing(('[EOS]', EOS_IDX), ('[BOS]', BOS_IDX))
# print(EXPR_TOKENIZER.get_vocab())


special_symbols = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '[MASK]', '[BOS]', '[EOS]']
PAD_IDX, SEP_IDX, CLS_IDX, UNK_IDX, MASK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3, 4, 5, 6
BASE_IDX, NUM_IDX = 7, len("0123456789+-=")

class ExprVocab:
    def __init__(self):
        self.id_map = {c:i for i, c in enumerate(special_symbols + list("0123456789+-=*/()xyz"))}
        self.token_map = {i:c for i, c in enumerate(special_symbols + list("0123456789+-=*/()xyz"))}

    def encode_batch(self, expr_str_l):
        max_len = max(len(s) for s in expr_str_l)
        token_id_l = []
        pad_l = []
        for s in expr_str_l:
            token_id_l.append([CLS_IDX] + [self.id_map[c] for c in s] + [EOS_IDX] + [PAD_IDX]*(max_len - len(s)))
            pad_l.append([0]*(len(s)+2) + [1]*(max_len - len(s)))
        return token_id_l, pad_l

    def get_vocab(self):
        return list(self.id_map.values())

EXPR_TOKENIZER = ExprVocab()
# print(EXPR_TOKENIZER.encode_batch(["1+2=3+0", "5+61=660000001"]))
