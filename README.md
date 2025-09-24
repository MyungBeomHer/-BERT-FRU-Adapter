## Speaker Recognition

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
#Frame recalibration unit adapter
class FRU_Adapter(nn.Module):
    def __init__(self,
                 channel = 197,
                 embded_dim = 1024,
                 Frame = 16,
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
        bt, n,d = x.shape
        x = rearrange(x, '(b t) n d-> (b n) t d', t = self.Frame, n = n, d = d)

        x = self.linear1(x) # bn t d
        x = self.ln(x) 

        _, _,down = x.shape

        x = rearrange(x, '(b n) t d-> b t (n d)', t = self.Frame, n = n, d = down)
        x = self.FRU(x)
        x = rearrange(x, 'b t (n d)-> (b n) t d', t = self.Frame, n = n, d = down)

        x = self.TFormer(x)
        x = self.linear2(x) # bn t d
        #bt n d
        x = rearrange(x, '(b n) t d-> (b t) n d', t = self.Frame, n = n, d = d)
        return x
```
[models_vit.py](models_vit.py)

- Benchmark (NER Dataset)

|Model|Params|MacroAvg F1 score|
|:------:|:------:|:---:|
|KoBERT|92.21M|0.8554|
|KoBERT+BiLSTM+CRF|95.75M|0.8659||
|**KoBERT+FRU-Adapter+CRF**|95.38M|**0.8703**||

### Reference Repo
- [NLP implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/docs/index.rst)
- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT)
- [Finetuning configuration from huggingface](https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_multiple_choice.py)
- [BERT Attention Visualization](https://github.com/jessevig/bertviz)

- 
