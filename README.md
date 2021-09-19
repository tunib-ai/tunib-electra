# TUNiB-Electra
  
We release several new versions of the [ELECTRA](https://arxiv.org/abs/2003.10555) model, which we name TUNiB-Electra. There are two motivations. First, all the existing pre-trained Korean encoder models are monolingual, that is, they have knowledge about Korean only. Our bilingual models are based on the balanced corpora of Korean and English. Second, we want new off-the-shelf models trained on much more texts. To this end, we collected a large amount of Korean text from various sources such as blog posts, comments, news, web novels, etc., which sum up to 100 GB in total.

You can use TUNiB-Electra with the Hugging Face [transformers](https://github.com/huggingface/transformers) library.  
  
### What's New:

- Sep 19, 2021 [Released a tech blog](https://tunib.notion.site/TECH-2021-09-18-TUNiB-Electra-3eba9f55859d4992a085a64c600dc150)
- Sep 17, 2021 [Released TUNiB-Electra](https://github.com/tunib-ai/tunib-electra). 
  
## How to use
  
You can use this model directly with [transformers](https://github.com/huggingface/transformers) library:
  
```python
from transformers import AutoModel, AutoTokenizer

# Small Model (Korean-English bilingual model)
tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-small')
model = AutoModel.from_pretrained('tunib/electra-ko-en-small')

# Base Model (Korean-English bilingual model)
tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-base')
model = AutoModel.from_pretrained('tunib/electra-ko-en-base')

# Small Model (Korean-only model)
tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-small')
model = AutoModel.from_pretrained('tunib/electra-ko-small')

# Base Model (Korean-only model)
tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-base')
model = AutoModel.from_pretrained('tunib/electra-ko-base')
```

### Tokenizer example

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-base')
>>> tokenizer.tokenize("tunib is a natural language processing tech startup.")
['tun', '##ib', 'is', 'a', 'natural', 'language', 'processing', 'tech', 'startup', '.']
>>> tokenizer.tokenize("튜닙은 자연어처리 테크 스타트업입니다.")
['튜', '##닙', '##은', '자연', '##어', '##처리', '테크', '스타트업', '##입니다', '.']
```
  
## Results on Korean downstream tasks
  
### Small Models
  
|                       |**# Params** |**Avg.**| **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |**Korean-Hate-Speech (Dev)**<br/>(F1)| 
|  :----------------:| :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :----------------: |
|***TUNiB-Electra-ko-small*** |   14M |  81.29|  **89.56**      |        84.98         |     72.85   |   77.08   |    78.76   | **94.98**  | 61.17 / 87.64  |  **64.50** |
|***TUNiB-Electra-ko-en-small*** |  18M |   81.44 | 89.28   |      85.15         |  75.75       | 77.06     | 77.61 | 93.79  | 80.55 / 89.77      |63.13 |
| [KoELECTRA-small-v3](https://github.com/monologg/KoELECTRA)    | 14M |  **82.58** | 89.36   |      **85.40**	     |    **77.45**    |    **78.60**    |       **80.79**      |     94.85    | **82.11 / 91.13**	|  63.07 | 

### Base Models
  
|                       |**# Params** |**Avg.**| **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |**Korean-Hate-Speech (Dev)**<br/>(F1)|
|  :----------------:| :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :----------------: |
|***TUNiB-Electra-ko-base*** |  110M | **85.99** |  90.95 |    87.63         |   **84.65**   | **82.27**   |    85.00   |  95.77 |   64.01 / 90.32   |71.40 |
|***TUNiB-Electra-ko-en-base*** |  133M |84.74 	|90.15      |        86.93         |    83.05      |  79.70    |  82.23 | 95.64  | 83.61 / 92.37     |67.86 |
| [KoELECTRA-base-v3](https://github.com/monologg/KoELECTRA)    |  110M | 85.92   |90.63   |      **88.11**	     |    84.45    |    82.24    |       **85.53**      |     95.25      | **84.83 / 93.45**	     |  67.61 |
| [KcELECTRA-base](https://github.com/Beomi/KcELECTRA) | 124M|  84.75     |**91.71**      |         86.90          |       74.80        |        81.65         |           82.65           |          **95.78**          |         70.60 / 90.11         | **74.49** |
| [KoBERT-base](https://github.com/SKTBrain/KoBERT)        |  90M  |   81.92       |  89.63        |         86.11          |       80.65        |        79.00         |           79.64           |            93.93            |         52.81 / 80.27         | 66.21 |
| [KcBERT-base](https://github.com/Beomi/KcBERT)         |   110M    |   79.79    | 89.62        |         84.34          |       66.95        |        74.85         |           75.57           |            93.93            |         60.25 / 84.39         |  68.77 |
| [XLM-Roberta-base](https://github.com/pytorch/fairseq/tree/master/examples/xlmr)   | 280M  | 83.03    |89.49        |         86.26          |       82.95        |        79.92         |           79.09           |            93.53            |         64.70 / 88.94         |  64.06  |


  
## Results on English downstream tasks
 
### Small Models
  
|                       |**# Params** | **Avg.** |**CoLA**<br/>(MCC) | **SST**<br/>(Acc) |MRPC<br/>(Acc)| **STS**<br/>(Spearman) | **QQP**<br/>(Acc) | **MNLI**<br/>(Acc) | **QNLI**<br/>(Acc) | **RTE**<br/>(Acc) | 
|  :----------------:| :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :---------------------------: |
|***TUNiB-Electra-ko-en-small*** |  18M | **80.44**  |	**56.76**       | 88.76       |   **88.73**      |  **86.12**     |  **88.66**  | 79.03   |  87.26    |**68.23** | 
|[ELECTRA-small](https://github.com/google-research/electra) | 13M |  79.71 | 	55.6      |     **91.1**            | 84.9|  84.6      |   88.0   | **81.6**  | **88.3**  |  63.6    | 
|[BERT-small](https://github.com/google-research/bert) |  13M |  74.06|	27.8      |      89.7           | 83.4|   78.8     |  87.0    | 77.6  |  86.4 | 61.8     | 

  
### Base Models
 
|                       |**# Params** | **Avg.** |**CoLA**<br/>(MCC) | **SST**<br/>(Acc) |MRPC<br/>(Acc)| **STS**<br/>(Spearman) | **QQP**<br/>(Acc) | **MNLI**<br/>(Acc) | **QNLI**<br/>(Acc) | **RTE**<br/>(Acc) | 
|  :----------------:| :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :---------------------------: |
|***TUNiB-Electra-ko-en-base***  | 133M |	 85.2| **66.29** |  91.86      |    **89.95**     | 89.67     |  **90.75** | 84.72  |    91.40 |**76.90**| 
|[ELECTRA-base](https://github.com/google-research/electra) | 110M |   **85.7** |	64.6     |     **96.0**           | 88.1|  **90.2**     |    89.5   |  **88.5**  |  **93.1**      |  75.2    | 
|[BERT-base](https://github.com/google-research/bert) | 110M |   80.8| 	52.1      |      93.5           |  84.8|    85.8     |  89.2   | 84.6        |   90.5       |  66.4    | 

 
## Pre-training data
 
- [***The Pile***](https://github.com/EleutherAI/the-pile)
- [***Reddit***](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit)
- [***OpenWebText***](https://github.com/jcpeterson/openwebtext)
- [***OpenSubtitles***](https://opus.nlpl.eu/OpenSubtitles-v2018.php)
- [***Modu Corpus***](https://corpus.korean.go.kr/)
- [***Namuwiki***](https://github.com/lovit/namuwikitext)
- [***KcBERT***](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_comments.html)
- [***KoWiki***](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/kowikitext.html)
- [***Korean Petitions***](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_petitions.html)
- [***AI Hub Translation***](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_parallel_koen_news.html)
- Etc.   

## Acknowledgement

The project was created with Cloud TPU support from the **Tensorflow Research Cloud (TFRC)** program. 
  
## Citation
  
If you find this code/model useful, please consider citing:
  
```
@misc{tunib-electra,
  author       = {Ha, Sangchun and Kim, Soohwan and Ryu, Myeonghyeon and
                  Keum, Bitna and Oh, Saechan and Ko, Hyunwoong and Park, Kyubyong},
  title        = {TUNiB-Electra},
  howpublished = {\url{https://github.com/tunib-ai/tunib-electra}},
  year         = {2021},
}
```
  
## License
  
`TUNiB-Electra` is licensed under the terms of the Apache 2.0 License.   
  
Copyright 2021 TUNiB Inc. http://www.tunib.ai All Rights Reserved.

