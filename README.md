# Assignment 3

## Object of this assignment

label이 전혀 없는 현실적인 상황을 다룬다. label없이 분류시스템을 구축한다.

1. part 1: generating silver label
   정답라벨이 없기 때문에 대체할 silver label을 생성해야 한다.
   1. lexical similarity
   1. embedding similarity
   1. ensemble
2. part 2: training with silver label
   1. self-training
   1. label embedding model
   1. consistency regularizaiton
   1. stabilization based on ensemble

## origin code analysis

### P3a.train-with-BERT-embedding-no-label.ipynb

#### code block 1

```python
pid2text = load_corpus(CORPUS_PATH) # load corpus

label2id = load_json(LABEL2ID_PATH)
id2label = load_json(ID2LABEL_PATH)
pid2label_test = load_json(PID2LABEL_TEST_PATH)

# loading pre-trained embeddings

corpus_data = torch.load(EMB_PATH) # {"ids": [...], "embeddings": Tensor}
pid_list = corpus_data["ids"]
pid2idx = {pid: i for i, pid in enumerate(pid_list)}
embeddings = corpus_data["embeddings"]

label_data = torch.load(LABEL_EMB_PATH)
label_emb = label_data["embeddings"].to(device)
```

1. pid2text: 제품 ID(pid)를 제품 텍스트(이름 + 설명)로 매핑 -> used
2. label2id: label을 label id로 매핑
3. id2label: label id를 label 이름으로 매핑
4. pid2label_test: test 제품 ID를 label ID 리스트로 매핑
5. corpus_data: torch.load() 로 로드한 사전 훈련된 BERT 임베딩 데이터
6. pid_list: corpus_data['ids']로 추출한 제품 ID 리스트. embeddings와 순서가 일치
7. pid2idx: 제품 ID를 embeddings 텐서의 인덱스로 매핑
8. embeddings: 모든 제품의 BERT 평균 임베딩 텐서. corpus_data["embeddings"]에서 추출. 각 행은 제품 하나, 열은 임베딩 차원. -> used
9. label_data: torch.load()로 로드한 카테고리 레이블의 BERT 임베딩 데이터
10. label_emb: label_data["embeddings"]를 GPU로 이동한 텐서. 각 행이 카테고리 레이블 하나, 열이 임베딩 차원. 레이블 임베딩 모델에서 사용. -> used
    > pid2text, label_emb로 lexical similarity를 계산하고, embeddings, label_emb로 embedding similarity를 계산하면 될듯.
