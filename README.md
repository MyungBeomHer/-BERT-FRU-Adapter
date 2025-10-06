## AI based korean Named Entity Recognizer

팀원 : [허명범](https://github.com/MyungBeomHer)

### 프로젝트 주제 
한국어 개체명 인식기

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Dataset
- [NER Dataset from 한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER)

### NER tagset
- 총 8개의 태그가 있음
    - PER: 사람이름
    - LOC: 지명
    - ORG: 기관명
    - POH: 기타
    - DAT: 날짜
    - TIM: 시간
    - DUR: 기간
    - MNY: 통화
    - PNT: 비율
    - NOH: 기타 수량표현
- 개체의 범주 
    - 개체이름: 사람이름(PER), 지명(LOC), 기관명(ORG), 기타(POH)
    - 시간표현: 날짜(DAT), 시간(TIM), 기간 (DUR)
    - 수량표현: 통화(MNY), 비율(PNT), 기타 수량표현(NOH)

## ➡️ Data Preparation
```bash
cd data_in/NER-master/
unzip 말뭉치\ -\ 형태소_개체명/.zip
```

### Requirements
```bash
pip install -r requirements.txt
```

### train
```bash
python train_bert_crf.py 
```

### Model
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
#KobertCRF + FRU-Adapter
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
```
[model/net.py](model/net.py)

- Benchmark (NER Dataset)

|Model|Params|MacroAvg F1 score|
|:------:|:------:|:---:|
|KoBERT|92.21M|0.8554|
|KoBERT+BiLSTM+CRF|95.75M|0.8659||
|**KoBERT+FRU-Adapter+CRF**|95.38M|**0.8703**||

### Reference Repo
- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT)
- [Finetuning configuration from huggingface](https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_multiple_choice.py)
- [SKTBrain KoBERT Error revise](https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf)
- [FRU-Adapter](https://github.com/SeoulTech-HCIRLab/FRU-Adapter)
