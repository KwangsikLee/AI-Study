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

### 8월 11일 ~ 8월22일

  - [키워드 검색한 뉴스요약](lectLanguageModel3/키워드_뉴스요약.ipynb) -키워드 기반 뉴스 요약 실습
gogamza/kobart-summarization 모델 사용
Chatbot에 키워드 검색하면 관련 뉴스를 crawling 하여 탐색후 뉴스 요약
~~~ python
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

### 8월 4일 ~ 8월15일


## 🛠️ 기술 스택
- **언어**: Python
- **ML/DL**: PyTorch, Transformers, Scikit-learn
- **데이터**: NumPy, Pandas, Matplotlib, Seaborn
- **웹**: Streamlit, BeautifulSoup, Selenium
- **NLP**: Hugging Face, KoNLPy, WordCloud