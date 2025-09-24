from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertModel, BertConfig
# from pytorch_pretrained_bert import BertModel, BertConfig
from TorchCRF import CRF
# from torchcrf import CRF
from einops import rearrange
import math

bert_config = {'attention_probs_dropout_prob': 0.1,
                 'hidden_act': 'gelu',
                 'hidden_dropout_prob': 0.1,
                 'hidden_size': 768,
                 'initializer_range': 0.02,
                 'intermediate_size': 3072,
                 'max_position_embeddings': 512,
                 'num_attention_heads': 12,
                 'num_hidden_layers': 12,
                 'type_vocab_size': 2,
                 'vocab_size': 8002}

class KobertBiLSTMCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertBiLSTMCRF, self).__init__()

        if vocab is None: # pretraining model 사용
            self.bert, self.vocab = get_pytorch_kobert_model()
        else: # finetuning model 사용           
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]

        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=config.dropout, batch_first=True, bidirectional=True)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_labels=num_classes)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=True):

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        attention_mask = input_ids.ne(self._pad_id).float()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            pack_padded_last_encoder_layer = pack_padded_sequence(last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, hc = self.bilstm(pack_padded_last_encoder_layer)
            outputs = pad_packed_sequence(outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ff(outputs)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

bert_config = {'attention_probs_dropout_prob': 0.1,
                 'hidden_act': 'gelu',
                 'hidden_dropout_prob': 0.1,
                 'hidden_size': 768,
                 'initializer_range': 0.02,
                 'intermediate_size': 3072,
                 'max_position_embeddings': 512,
                 'num_attention_heads': 12,
                 'num_hidden_layers': 12,
                 'type_vocab_size': 2,
                 'vocab_size': 8002}


class FRU_Adapter(nn.Module):
    def __init__(self,
                 channel = 197,
                 embded_dim = 1024,
                 Frame = 30,
                 hidden_dim = 128):
        super().__init__()

        self.Frame = Frame

        self.linear1 = nn.Linear(embded_dim ,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,embded_dim)

        self.T_linear1 = nn.Linear(Frame, Frame)
        self.softmax = nn.Softmax(dim=1)
        self.ln = nn.LayerNorm(hidden_dim)
        
        self.TFormer = TemporalTransformer(frame=Frame,emb_dim=hidden_dim)

    #Frame recalibration unit
    def FRU(self, x):
        x1 = x.mean(-1).flatten(1) # bn t 
        x1 = self.T_linear1(x1) # bn t
        x1 = self.softmax(x1).unsqueeze(-1) #bn t 1

        x = x * x1 #bn t d
        return x 
    
    def forward(self, x):
        #x = bt N D 
        b,t,d = x.shape
        #bN t D ,FC -> GAP(D) 이후 TSE-> t-former(only D) -> FC
        # x = rearrange(x, '(b t) n d-> (b n) t d', t = self.Frame, n = n, d = d)

        x = self.linear1(x) # b t d
        x = self.ln(x) 

        _, _,down = x.shape
        # x = rearrange(x, '(b n) t d-> b t (n d)', t = self.Frame, n = n, d = down)
        x = self.FRU(x)
        # x = rearrange(x, 'b t (n d)-> (b n) t d', t = self.Frame, n = n, d = down)

        x = self.TFormer(x)
        x = self.linear2(x) # bn t d
        #bt n d
        # x = rearrange(x, '(b n) t d-> (b t) n d', t = self.Frame, n = n, d = d)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, 
                 frame = 16,
                 #channel = 8,
                 emb_dim = 128,
                 ):
        super().__init__()
        
        self.proj_Q = nn.Linear(emb_dim,emb_dim)
        self.proj_K = nn.Linear(emb_dim,emb_dim)
        self.proj_V = nn.Linear(emb_dim,emb_dim)
        self.proj_output = nn.Linear(emb_dim,emb_dim)
        
        self.norm = nn.LayerNorm(emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        #B C T H W 
        _,_,E = x.shape
            
        x1 = self.norm(x) 

        q = self.proj_Q(x1)
        k = self.proj_K(x1)
        v = self.proj_V(x1)

        q_scaled = q * math.sqrt(1.0 / float(E))

        attn_output_weights = q_scaled @ k.transpose(-2, -1)
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output = attn_output_weights @ v 
        attn_output = self.proj_output(attn_output) #B T E  where E = C * H * W
        attn_output = attn_output + x 

        return attn_output 

class KobertCRF(nn.Module):
    """ KoBERT with CRF FRU-Adapter"""
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRF, self).__init__()

        if vocab is None:
            self.bert, self.vocab = get_pytorch_kobert_model()
        else:
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab

        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_labels=num_classes)
        self.pad_id = getattr(config, "pad_id", 1)  # 기본 1

        self.tsea_blocks = nn.ModuleList([
            FRU_Adapter(embded_dim=768) for _ in range(12)
        ])

        # head_mask = [None] * self.bert.config.num_hidden_layers

        for param in self.bert.encoder.parameters():
           param.requires_grad = False



    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float() # B, 30

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        #outputs: (last_encoder_layer, pooled_output, attention_weight) 
        # for i, layer_module in enumerate(self.bert.encoder.layer):
        hidden_states  = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids) # B 30 768
        # head_mask = [None] * self.bert.config.num_hidden_layers
        
        for i, blk in enumerate(self.bert.encoder.layer):
            hidden_states = blk(hidden_states,attention_mask)#,head_mask[i])
            hidden_states = hidden_states[0] if isinstance(hidden_states, (tuple, list)) else hidden_states
            hidden_states = hidden_states + self.tsea_blocks[i](hidden_states)
        
        last_encoder_layer = hidden_states #outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)
        mask = input_ids.ne(self.pad_id)   # dtype=bool
        max_len = input_ids.size(1)
        pad_val = self.pad_id  # = 1

        def _pad_paths(paths):
            # paths: List[List[int]] (batch 크기)
            out = []
            for p in paths:
                if len(p) < max_len:
                    p = p + [pad_val] * (max_len - len(p))
                out.append(p)
            return torch.tensor(out, device=input_ids.device, dtype=torch.long)

        if tags is not None:
            # log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            # sequence_of_tags = self.crf.decode(emissions, mask=mask)
            log_likelihood = self.crf(emissions, tags, mask=mask)
            sequence_of_tags = self.crf.viterbi_decode(emissions, mask=mask)
            sequence_of_tags = _pad_paths(sequence_of_tags) #---
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, mask=mask)
            sequence_of_tags = _pad_paths(sequence_of_tags) #---
            return sequence_of_tags

