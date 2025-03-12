import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):

    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()
        
        self.l1 = nn.Linear(in_features=q_input_dim, out_features=kq_dim)
        self.l2 = nn.Linear(in_features=cand_input_dim, out_features=kq_dim)
        self.l3 = nn.Linear(in_features=cand_input_dim, out_features=v_dim)

        self.kq_dim = kq_dim


    def forward(self, hidden, encoder_outputs):
        
        t1 = self.l1(hidden)
        t2 = self.l2(encoder_outputs)
        t3 = self.l3(encoder_outputs)
        
        t1 = t1.unsqueeze(1)

        scores = torch.bmm(t1, t2.transpose(1,2))
        scores = scores / (self.kq_dim ** 0.5)

        alpha = torch.softmax(scores, dim=-1)

        attended_val = torch.bmm(alpha, t3)
        attended_val = attended_val.squeeze(1)

        alpha = alpha.squeeze(1)
        return attended_val, alpha



class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        
    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], self.v_dim) ).to(hidden.device)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device), dim=-1)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.embedding = nn.Embedding(num_embeddings=src_vocab_len, embedding_dim=emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_lens):
        
        embeded = self.embedding(src)
        droped = self.dropout(embeded)
        output, _ = self.gru(droped)

        forward_output = output[:, :, :self.enc_hid_dim]
        backward_output = output[:, :, self.enc_hid_dim:]

        word_representations = torch.cat((forward_output, backward_output), dim=2)

        batch = torch.arange(output.shape[0]).to(src.device)
        last_forward = forward_output[batch, src_lens - 1]
        first_back = backward_output[:, 0, :]

        sentence_rep = torch.cat((last_forward, first_back), dim=1)
        # print(f"word rep shape: {word_representations.shape}")
        # print(f"sentence rep shape: {sentence_rep.shape}")
        return word_representations, sentence_rep


class Decoder(nn.Module):
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()

        self.attention = attention
        self.embedding = nn.Embedding(num_embeddings=trg_vocab_len, embedding_dim=emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=dec_hid_dim, bidirectional=False, batch_first=True)
        self.dec_linear = nn.Linear(in_features=dec_hid_dim, out_features=dec_hid_dim)
        self.gelu = nn.GELU()
        self.trg_linear = nn.Linear(in_features=dec_hid_dim, out_features=trg_vocab_len)



    def forward(self, input, hidden, encoder_outputs):
        # print(f"Input shape: {input.shape}")
        droped = self.dropout(self.embedding(input))
        droped = droped.unsqueeze(1)
        # print(f"Hidden shape before GRU: {hidden.shape}")
        # print(f"Dropout output shape: {droped.shape}")
        output, hn = self.gru(droped, hidden.unsqueeze(0))
        
        hn = hn.squeeze(0)

        attended, alphas = self.attention(hn, encoder_outputs)

        new_hidden = hn + attended

        out = self.dec_linear(new_hidden)
        out = self.gelu(out)
        out = self.trg_linear(out)

        return new_hidden, out, alphas

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention, dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim*2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2*enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2*enc_hidden_dim, dec_hidden_dim, kq_dim)

        
        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)
        



    def translate(self, src, src_lens, sos_id=1, max_len=50):
        
        #tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)*sos_id

        word_rep, sentence_rep = self.encoder(src, src_lens)
        hidden = self.enc2dec(sentence_rep)

        for t in range(max_len):
            hidden, output, attn_weights = self.decoder(input_words, hidden, word_rep)
            predicted_word = output.argmax(dim=1)
            outputs[:, t] = predicted_word
            attns[:, t, :] = attn_weights
            input_words = predicted_word

        return outputs, attns
        

    def forward(self, src, trg, src_lens):

        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device)

        word_rep, sentence_rep = self.encoder(src, src_lens)
        # print("BxTx2*enc_hid_dim tensor word_representations: ", word_rep.shape)
        # print("sentence_rep should be a Bx2*enc_hid_dim tensor", sentence_rep.shape)
        hidden = self.enc2dec(sentence_rep)
        # print("trg.shape[1]: ", trg.shape[1])
        for t in range(1, trg.shape[1]-1):
            input_word = trg[:, t-1]
            hidden, output, _ = self.decoder(input_word, hidden, word_rep)
            outputs[:, t-1, :] = output

        return outputs