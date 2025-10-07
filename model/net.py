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
        #x = b t D         
        x = self.linear1(x) # b t d
        x = self.ln(x) 

        x = self.FRU(x)
        
        x = self.TFormer(x)
        x = self.linear2(x) # b t d
       
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

        # for param in self.bert.encoder.parameters():
        #    param.requires_grad = False
        # for p in self.bert.embeddings.parameters():
        #     p.requires_grad = False

        # for i, layer in enumerate(self.bert.encoder.layer):
        #     for name, param in layer.named_parameters():
        #         if "LayerNorm" in name:
        #             param.requires_grad = True  # 나머지는 False 유지

    def forward(self, input_ids, token_type_ids=None, tags=None):
        # --- 1) BERT attention mask (2D -> extended additive mask) ---
        pad = self.vocab.token_to_idx[self.vocab.padding_token]
        mask_2d = input_ids.ne(pad).to(dtype=self.bert.embeddings.word_embeddings.weight.dtype)  # [B, L]
        extended_attention_mask = self.bert.get_extended_attention_mask(
            mask_2d, mask_2d.shape, device=input_ids.device
        )  # [B,1,1,L], tokens: 0.0, pads: -10000.0

        # --- 2) 임베딩 + (동결된) 인코더 + FRU 병렬잔차 ---
        hidden_states = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)  # [B,L,768]

        for i, encoder_layer in enumerate(self.bert.encoder.layer):
            prev = hidden_states
            out = encoder_layer(hidden_states=hidden_states, attention_mask=extended_attention_mask)
            x = out[0] if isinstance(out, (tuple, list)) else out
            x = x + self.tsea_blocks[i](prev)  # FRU Adapter
            # x = x + self.tsea_blocks[i](x)  # FRU Adapter
            hidden_states = x

        last_encoder_layer = self.dropout(hidden_states)
        emissions = self.position_wise_ff(last_encoder_layer)  # [B,L,num_classes]

        # --- 3) CRF용 mask (bool, [B,L]) ---
        crf_mask = input_ids.ne(self.pad_id)  # True=유효토큰, False=패딩
    
        max_len = input_ids.size(1)   # y_real과 동일한 목표 길이
        pad_val = self.pad_id         # 패딩값 (어차피 acc 계산에서 pad는 마스크됨)

        def _pad_paths(paths, tgt_len, pad_val):
            out = []
            for p in paths:
                if len(p) < tgt_len:
                    p = p + [pad_val] * (tgt_len - len(p))
                else:
                    p = p[:tgt_len]
                out.append(p)
            return out

        if tags is not None:
            log_likelihood = self.crf(emissions, tags, mask=crf_mask.to(torch.uint8))
            seq = self.crf.viterbi_decode(emissions, mask=crf_mask.to(torch.uint8))
            seq = _pad_paths(seq, max_len, pad_val)                       # ★ 패딩
            sequence_of_tags = torch.tensor(seq, device=input_ids.device) # 텐서로 변환
            return log_likelihood, sequence_of_tags
        else:
            seq = self.crf.viterbi_decode(emissions, mask=crf_mask.to(torch.uint8))
            seq = _pad_paths(seq, max_len, pad_val)                       # ★ 패딩
            sequence_of_tags = torch.tensor(seq, device=input_ids.device)
            return sequence_of_tags

