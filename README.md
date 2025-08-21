# AI-Study 📚

AI/ML 학습을 위한 개인 스터디 저장소입니다. 기초 Python부터 최신 딥러닝, 자연어처리까지 체계적으로 학습한 내용을 정리하고 있습니다.

## 📋 프로젝트 구조

### 🐍 기초 학습 모듈
- **lectBasic/**: Python 기초, 데이터베이스, Streamlit 기본기
- **lectCrawling/**: 웹 크롤링 (BeautifulSoup, Selenium)
- **lectAPI/**: API 연동 실습 (World Bank, Hacker News 등)

### 📊 데이터 분석 & ML
- **lectAnalysis/**: NumPy, Pandas 데이터 분석 기초
- **lectMLDL/**: 전통적 머신러닝 (KNN, 결정트리, 선형회귀)
- **lectUnsupervised/**: 비지도학습 (클러스터링, PCA, t-SNE)

### 🧠 딥러닝 & AI
- **lectDL_NN/**: PyTorch 기반 딥러닝 (MLP, CNN, RNN/LSTM)
- **lectPytorch/**: PyTorch 심화 학습
- **lectVision/**: 컴퓨터 비전 (YOLO 객체탐지)

### 🤖 자연어처리 & 언어모델
- **lectLanguageModel1/**: NLP 기초 (토크나이징, TF-IDF, 임베딩)
- **lectLanguageModel2/**: 고급 NLP (BERT, Attention, 감정분석)
- **lectLanguageModel3/**: 최신 언어모델 (GPT-2, 파인튜닝, 챗봇)

### 🔍 고급 주제
- **RAG_Study/**: RAG(검색증강생성) 연구



## 📝 학습 일지

### Language Model 8월 11일 ~ 8월22일


- [다국어 번역기](lectLanguageModel3/다국어번역기.ipynb) - 다국어 번역기.  
번역 모델 : facebook/m2m100_418M



- [지속적인 수집 및 학습](lectLanguageModel3/데이터수집_청킹_증분.ipynb) - simple 학습 daemon   
AutoUpdateScheduler : simple daemon 실시간 데이터 수집 및 학습 

- [chunking & 요약정리](lectLanguageModel3/Chunks.ipynb)
tokenizer로 token 변환후 슬라이딩 윈도우 청킹.  
요약 : Map Reduce 구조

- [chunking & embedding](lectLanguageModel3/chunking_gradio.ipynb) - 청킹, Gradio UI  
토크나이저 모델 : "klue/bert-base"  
simple chunking

- [embedding & 검색](lectLanguageModel3토링_예제.ipynb) - 현상 -> 원인, 부품, 해결 방법 검색

    임베딩 모델 : "jhgan/ko-sroberta-multitask"  # 한국어 SBERT, CPU에서도 빠름
    ~~~
    sims = (self.alias_emb @ q).tolist() 
    ~~~
    1. 행렬 곱셈 연산 (@)
    - @ 연산자는 Python 3.5+에서 도입된 행렬 곱셈 연산자
    - NumPy 배열 간의 내적(dot product) 계산
    - self.alias_emb @ q는 np.dot(self.alias_emb, q)와 동일


- [PDF embedding & 검색](lectLanguageModel3/사용설명서_pdf_챗봇.ipynb) - pdf embedding, 

    simple chunking.  
    임베딩(문장 벡터) & FAISS 인덱스 구축(cosine 유사도)  
    Gradio 간단 챗봇 UI

- [소설 작성](lectLanguageModel3/소설_개선.ipynb) - prompt 기반 소설 작성. 

    Qwen/Qwen2.5-3B-Instruct 모델 사용. 여러 모델들을 사용하여 테스트 하기 용이.  
    outline - 초안 - 검토 단계를 단계별 결과를 이용하여 발전시키는 방식. 결과 후처리 샘플.


- [키워드 검색한 뉴스요약](lectLanguageModel3/키워드_뉴스요약.ipynb) -키워드 기반 뉴스 요약 실습.  
gogamza/kobart-summarization 모델 사용.  
Chatbot에 키워드 검색하면 관련 뉴스를 crawling 하여 탐색후 뉴스 요약
    ~~~ python
    # 요약 모델
    pipeline( "summarization",model="gogamza/kobart-summarization",
                    tokenizer="gogamza/kobart-summarization",
                    device=0 if torch.cuda.is_available() else -1
                )
    ~~~

- [뉴스 감성 분석](lectLanguageModel3/뉴스_긍정_부정_GPT2.ipynb)  
kogpt2 모델을 Classification 모델로도 활용할 수 있다  
    ~~~
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", num_labels=num_labels)
    model.cuda()
    ~~~

### Language Model 8월 4일 ~ 8월15일
- [감성 분석](lectLanguageModel2/Attension_감정분석_어텐션.ipynb) - Graph Attention Layer  
GraphEmotionNetwork 클래스는 BERT와 그래프 신경망(GNN)을 결합해 감정 분석을 수행하는 모델 

- [기계 번역](lectLanguageModel2/GRU_기계번역.ipynb) - GRU (Gated Recurrent Unit)  
pyTorch,  encoder & decoder (GRU model base)

- [GRU로 한글 의도 분류 (Intent Classification)](lectLanguageModel2/GRU_의도분류.ipynb) - GRU(Gated Recurrent Unit)  
tensorflow,  Bidirectional GRU for better context understanding

- [품사 태깅 예제](lectLanguageModel2/NER_품사_태깅.ipynb) - LSTM & bidirectional
pyTorch, 신경망모델 구축


### Language Model 8월 4일 ~ 8월15일

- [N-gram 모델](lectLanguageModel1/2_gram.ipynb) - 바이그램 확률

- [BPE 모델](lectLanguageModel1/BPE_Unigram.ipynb)    
sentencepiece 사용

- [TF-IDF](lectLanguageModel1/TF_IDF_영화추천(불용제거).ipynb) -  
Okt, mecab 형태소 분석


### Neural Network Model 


### Machine Learning




## 🛠️ 기술 스택
- **언어**: Python
- **ML/DL**: PyTorch, Transformers, Scikit-learn
- **데이터**: NumPy, Pandas, Matplotlib, Seaborn
- **웹**: Streamlit, BeautifulSoup, Selenium
- **NLP**: Hugging Face, KoNLPy, WordCloud