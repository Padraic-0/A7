import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import spacy
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
  def __init__(self, corpus, tokenizer):
    self.tokenizer = tokenizer
    self.word2idx, self.idx2word = self.build_vocab(corpus)

  def __len__(self):
    return len(self.word2idx)
  
  def text2idx(self, text):
    tokens = [str(x).strip().lower() for x in self.tokenizer(text)]
    return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['<UNK>'] for t in tokens]

  def idx2text(self, idxs):
    return [self.idx2word[i] if i in self.idx2word.keys() else '<UNK>' for i in idxs]


  def build_vocab(self,corpus):
    raw_tokens = [str(x).strip().lower() for x in self.tokenizer(" ".join(corpus))]
    cntr = Counter(raw_tokens)
    tokens = [t for t,c in cntr.items() if c >= 2]
    word2idx = {t:i+4 for i,t in enumerate(tokens)}
    idx2word = {i+4:t for i,t in enumerate(tokens)}
    
    word2idx['<PAD>'] = 0  #add padding token
    idx2word[0] = '<PAD>'

    word2idx['<SOS>'] = 1  #add padding token
    idx2word[1] = '<SOS>'

    word2idx['<EOS>'] = 2  #add padding token
    idx2word[2] = '<EOS>'

    word2idx['<UNK>'] = 3  #add padding token
    idx2word[3] = '<UNK>'
    

    return word2idx, idx2word

class Multi30kDatasetEnDe(Dataset):

  def __init__(self,split="train", vocab_en = None, vocab_de = None):

    dataset = load_dataset("bentrevett/multi30k", split=split)
    self.data_en = [x['en'] for x in dataset]
    self.data_de = [x['de'] for x in dataset]

    if vocab_en == None:
      self.vocab_en = Vocabulary(self.data_en, spacy.load('en_core_web_sm').tokenizer)
      self.vocab_de = Vocabulary(self.data_de, spacy.load('de_core_news_sm').tokenizer)
    else:
      self.vocab_en = vocab_en
      self.vocab_de = vocab_de

  def __len__(self):
    return len(self.data_en)

  def __getitem__(self, idx):
    numeralized_en = [self.vocab_en.word2idx['<SOS>']]+self.vocab_en.text2idx(self.data_en[idx])+[self.vocab_en.word2idx['<EOS>']]
    numeralized_de = self.vocab_de.text2idx(self.data_de[idx])
    return torch.tensor(numeralized_de),torch.tensor(numeralized_en)
  
  @staticmethod
  def pad_collate(batch):
    xx = [ele[0] for ele in batch]
    yy = [ele[1] for ele in batch]
    x_lens = torch.LongTensor([len(x)-1 for x in xx])
    y_lens = torch.LongTensor([len(y)-1 for y in yy])

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

def getMulti30kDataloadersAndVocabs(batch_size=128):
  multi_train = Multi30kDatasetEnDe(split="train")
  multi_val = Multi30kDatasetEnDe(split="validation", vocab_en=multi_train.vocab_en, vocab_de=multi_train.vocab_de)
  multi_test = Multi30kDatasetEnDe(split="test",  vocab_en=multi_train.vocab_en, vocab_de=multi_train.vocab_de)

  collate = Multi30kDatasetEnDe.pad_collate
  train_loader = DataLoader(multi_train, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=collate, drop_last=True)
  val_loader = DataLoader(multi_val, batch_size=batch_size, num_workers=8,  shuffle=False, collate_fn=collate)
  test_loader = DataLoader(multi_test, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate)

  return train_loader, val_loader, test_loader, {"en":multi_train.vocab_en, "de":multi_train.vocab_de}
