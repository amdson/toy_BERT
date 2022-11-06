
from collections import OrderedDict
import tokenizers
from tokenizers import BertWordPieceTokenizer
EXPR_TOKENIZER = BertWordPieceTokenizer(unk_token='[UNK]', sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]', mask_token='[MASK]')
EXPR_TOKENIZER.enable_padding()
special_symbols = ['[UNK]', '[SEP]', '[CLS]', '[PAD]', '[MASK]', '[BOS]', '[EOS]']
EXPR_TOKENIZER.add_special_tokens(special_symbols)
EXPR_TOKENIZER.add_tokens(list("0123456789+-*/()xyz"))
UNK_IDX, SEP_IDX, CLS_IDX, PAD_IDX, MASK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3, 4, 5, 6
EXPR_TOKENIZER.post_processor = tokenizers.processors.BertProcessing(('[EOS]', EOS_IDX), ('[BOS]', BOS_IDX))
# print(EXPR_TOKENIZER.get_vocab())
