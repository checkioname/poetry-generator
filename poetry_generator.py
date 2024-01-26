import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

#Carregando nossos dados
a = 'data/txt/poesia/almada.txt'
with open(a,'r') as file:
    data = file.read()
    #print(data)

corpus = data.lower().split("\n")
#separando o texto para apenas a parte relevante 
# for i,obj in enumerate(corpus):
#     print(i,obj)
corpus = corpus[99:]
#print(corpus)
pt = spacy.load('pt_core_news_sm')


padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 10
xlmr_spm_model_path = ""

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)


from torch.utils.data import DataLoader