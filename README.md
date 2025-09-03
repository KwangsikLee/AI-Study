# AI-Study 📚

AI/ML 학습을 위한 개인 스터디 저장소입니다. 기초 Python부터 데이터 분석, 최신 딥러닝, 자연어처리까지 체계적으로 학습한 내용을 정리하고 있습니다.

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
- **lectDL_NN/**: PyTorch 기반 딥러닝 (MLP, CNN, RNN/LSTM, 앙상블)
- **lectPytorch/**: PyTorch 심화 학습
- **lectVision/**: 컴퓨터 비전 (YOLO 객체탐지)

### 🤖 자연어처리 & 언어모델
- **lectLanguageModel1/**: NLP 기초 (토크나이징, TF-IDF, 임베딩)
- **lectLanguageModel2/**: 고급 NLP (BERT, Attention, 감정분석)
- **lectLanguageModel3/**: 다양한 언어모델 (GPT-2, 파인튜닝, 챗봇)
- **lectOpenAI/**: OpenAI의 API 

### 🔍 LLM
- **RAG_Study/**: RAG(검색증강생성) 연구 ([README.md](./RAG_Study/README.md)) 
- **Fine Tuning/**: Fine Tuning 연구

### VLM
- **VLM/**: VLM 연구 ([README.md](./VLM_Study/README.md)) 


## 📝 학습 일지
최근 학습순으로 정리

### Fine-Tuning 

- [Fine_Tuning_Demo_Unsloth](fine_tuning/Fine_Tuning_Demo_Unsloth/Fine_Tuning.ipynb) - Fine-Tuning LLMs with Unsloth   
    LoRA adapters 방식의 fine tuing : unsloth 를 이용하여 빠른 Fine-Turing  
    gguf 모델 저장 : base 모델과 합쳐진 모델로 저장

- [LLM_Fine Tuning Workflow](fine_tuning/qwen_lora_finetuning/LLM_FineTuningWorkflow.ipynb)  - PEFT LoRA  
    PEFT(Parameter-Efficient Fine-Tuning):  대규모 언어 모델(LLM)을 파인튜닝할 때, 모델 전체 파라미터를 학습하지 않고 일부만 학습해서 효율을 높이는 기법.  

    Qwen/Qwen2.5-3B-Instruct 모델을 LoRA 기반 fine-tuning  
    fine_tuned model : Adapter layer 저장. 

### [LangChain & Agent](lectLangSmith/)
- [newstool](lectLangSmith/11_langChain_Tool_1.ipynb) 
- [PythonREPLTool 파이썬 코드 실행툴](lectLangSmith/12_langChain_Tool_2.ipynb) 
- [with DALL-E API 이미지 생성](lectLangSmith/13_langChain_Tool_3.ipynb) 

### [LangSmith & RAG](lectLangSmith/)
- [RAG 평가 시스템 sample ](lectLangSmith/09_ev_ex4.ipynb) 
    LLM-as-Judge 평가 시스템 구축:   
    Chain of Thought 평가, 
    Pairwise Comparison, 
    Multi-aspect 평가 등
    3가지 고도화된 평가 방법론을 통해 LLM 답변 품질을 다각도로 측정하는 커스텀 평가자 시스템 개발


- [RAG 뉴스프로젝트 & LangSmith ](lectLangSmith/06_RAG_뉴스프로젝트.ipynb)  RAG 기반 챗봇
    ####  키워드 검색 시스템
    **6단계 다층 검색 전략으로 검색 성능 향상!**
    1. **BM25 원본 검색**: 입력한 키워드 그대로 검색
    2. **형태소 분석 검색**: 한국어 형태소 분석 후 핵심 키워드 추출하여 검색
    3. **개별 키워드 검색**: 추출된 각 키워드로 개별 검색
    4. **메타데이터 키워드 매칭**: 문서에 미리 추출된 키워드와 매칭
    5. **제목 우선 검색**: 뉴스 제목에서 키워드 매칭 (높은 점수)
    6. **콘텐츠 부분 매칭**: 본문 내 키워드 빈도 기반 검색

    ####  다중 임베딩 모델 지원
    - **ko-sroberta-multitask**: 한국어 특화 SentenceTransformer (300MB) ⭐ 추천
    - **paraphrase-multilingual**: 다국어 경량 모델 (420MB)
    - **distiluse-multilingual**: 다국어 고속 처리 (540MB)
    - **ko-electra**: 한국어 경량 모델 (50MB)
    - **xlm-r-100langs**: 100개 언어 지원 (1.1GB)

    ####  Reranker 기술 적용
    - **cross-encoder-ms-marco**: MS MARCO 경량 CrossEncoder (80MB) ⭐ 기본
    - **cross-encoder-multilingual**: 다국어 지원 (400MB) ⭐ 추천
    - **bge-reranker**: BGE Reranker (400MB)
    - **none**: Reranking 비활성화

    ####  4가지 고급 검색 방법
    - **Hybrid**: BM25 + 벡터 검색 앙상블 (균형잡힌 성능)
    - **Similarity**: 의미 기반 벡터 검색 (문맥 이해)
    - **Keyword**: 6단계 다층 키워드 검색 ⭐
    - **Fusion**: 다중 방법 융합 (최고 성능, 느림)


### [LangChain & RAG](lectLangChain/)

- [RAG & Chat bot ](lectLangChain/09_langchain_chatbot.ipynb)  RAG 기반 챗봇
    아래 내용들이 종합적으로 적용된 코드.  
    사용자 정의 re-rank Retriver 생성 (CrossEncoderRetriever)
    ~~~ python
    class CrossEncoderRetriever(BaseRetriever):
    ~~~

- [RAG & Chat bot2 ](lectLangChain/17_gpt5_리랭커추가.ipynb)  FlagReranker 
    아래 내용들이 종합적으로 적용된 코드.  
    FlagReranker
    ~~~ python
    from FlagEmbedding import FlagReranker
    ~~~    

- [LongContext Reorder ](lectLangChain/11_LongContextReorder.ipynb) Retriever의 결과를 재배치   
    긴 context의 경우 중간 부분의 context는 LLM에서 무시되는 경향(Lost in the Middle 문제).   
    중요한 것의 위치를 재조정

- [Parent Document Retriever](lectLangChain/12_Parent_Document_Retriever.ipynb) 이중 분할 구조로 검색 정확도와 컨텍스트 풍부함을 동시 확보    
    질문 → vectorstore(자식 검색) → 관련 부모 doc_id 찾기 → docstore에서 부모 청크 꺼내 반환

- [MultiVector Retriever](lectLangChain/13_MultiVectorRetriever.ipynb) Parent Document Retriever 의 확장    
    **MultiVectorRetriever**

- [Retriever + Re-ranker](lectLangChain/16_Cross_Encoder_Reranker.ipynb) Retriever + Re-ranker    
    vector search: top_k initial = 50 → re-rank to top_n = 3–7


- [Ensemble Retriever](lectLangChain/10_langchain_검색.ipynb)  
    Ensemble Retriever (BM25+FAISS)   
    BM25Retriever   
    BM25 (Best Matching 25) - 키워드 기반 검색 
    * 참고 : [with 한글형태소분석](lectLangChain/15_BM25Retriever_한글_형태소.ipynb)

- [LangChain History & Langchain log](lectLangChain/01_langchain_basic.ipynb) -  History, ConsoleCallbackHandler  
    **ConsoleCallbackHandler** callback.  
    ~~~ python
    result = llm_chain.invoke({"who":"이순신 장군"},
                            config={"callbacks": [ConsoleCallbackHandler()]})
    ~~~

- [LangChain LCEL](lectLangChain/01_langchain_basic.ipynb) -  LCEL  
    **LCEL** chaining.  
    ~~~
    chain = prompt | llm | output_parser
    ~~~

- [LangChain](lectLangChain/03_langchain_2.ipynb) -  LangChain, Embedding, VectorDB   
    vector db: Chroma.  
    embedding : OpenAIEmbeddings. 
    splitter : CharacterTextSplitter. 

- [Chunking](lectLangChain/05_langchain_chunking.ipynb) - SemanticChunker...  


### OpenAI ChatGPT 활용 

- [OpenAI Function](lectOpenAI/06_함수.ipynb) - Tool Calling(Function Calling)    
    OpenAI의 Tool Calling : 외부 함수나 API, DB 등을 자동으로 호출할 수 있도록 해주는 기능   
    단순 텍스트 응답 → 실행 가능한 에이전트로 바꿔주는 기능    
    활용 예:  
    - 실시간 데이터 가져오기  
    - 데이터베이스 조회
    - 계산/시뮬레이션 처리
    - 문서/파일 처리
    - 외부 서비스 제어
    - Multi step workflow


- [다국어 번역기](lectOpenAI/프롬프팅_cot.ipynb) - 프롬프팅 기법    
    Few-shot : 단 몇 개의 예시(examples)를 프롬프트(지시문)에 포함하여 제공하는 방법(In-context Learning)

    | 개념              | 설명                                                                 | 예시 (감정 분석) |
    |-------------------|----------------------------------------------------------------------|------------------|
    | **Zero-shot (제로샷)** | 예시 없이, 오직 지시만으로 작업을 요청. 모델의 기존 지식에 전적으로 의존. | "이 문장은 긍정이야 부정이야?" |
    | **One-shot (원샷)**    | 단 하나의 예시를 제공. 원하는 출력 형식을 명확히 할 때 유용.            | 입력: "배송 빨라요." -> 긍정<br>입력: "별로네요." -> ? |
    | **Few-shot (퓨샷)**    | 몇 개(2~5개)의 예시를 제공. 더 복잡한 패턴이나 뉘앙스를 학습시키기에 효과적. | (위의 구체적인 예시와 동일) |
    | **Fine-tuning (파인튜닝)** | 수백~수천 개 이상의 많은 데이터를 사용해 모델의 가중치(weights)를 직접 업데이트. 모델을 특정 작업에 깊게 전문화시키는 과정. | 감정분석 데이터셋으로 모델을 재훈련 |


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
**GPT-2 (Generative Pre-trained Transformer 2)**: 자동회귀 기반 생성형 언어모델  
- 특징: 이전 토큰들을 기반으로 다음 토큰을 순차적으로 예측  
- 구조: Transformer 디코더만을 사용한 단방향 언어모델  
- 사전훈련: 대량의 텍스트 데이터로 다음 단어 예측 학습  
- 활용: 텍스트 생성, 분류, 요약 등 다양한 NLP 태스크  
    ~~~
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", num_labels=num_labels)
    model.cuda()
    ~~~

### Language Model 8월 4일 ~ 8월15일
- [감성 분석](lectLanguageModel2/Attension_감정분석_어텐션.ipynb) - Graph Attention Layer  
**BERT (Bidirectional Encoder Representations from Transformers)**: 양방향 인코더 기반 사전훈련 언어모델  
    - 특징: 문맥을 양방향으로 이해하여 더 정확한 언어 표현 학습  
    - 구조: Transformer 인코더 스택으로 구성  
    - 사전훈련: Masked Language Model + Next Sentence Prediction  
    - 활용: 감정분석, 질의응답, 개체명인식 등 다양한 NLP 태스크에 fine-tuning 

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

- [MLP 분류](lectDL_NN/아이리스_MLP.ipynb) - 다층퍼셉트론으로 아이리스 분류  
**MLP (Multi-Layer Perceptron)**: 입력층, 은닉층, 출력층으로 구성된 완전연결 신경망  
    - 구조: 각 뉴런이 다음 층의 모든 뉴런과 연결  
    - 활성화함수: ReLU, Sigmoid, Tanh 등 사용  
    - 용도: 분류, 회귀 등 다양한 지도학습 문제 해결  

- [CNN 이미지 분류](lectDL_NN/CNN_음식_칼로리.ipynb) - CNN으로 음식 칼로리 예측  
**CNN (Convolutional Neural Network)**: 이미지 처리에 특화된 합성곱 신경망  
    - 핵심 구성: 합성곱층(Convolution), 풀링층(Pooling), 완전연결층  
    - 특징: 지역적 특성 추출, 위치 불변성, 매개변수 공유  
    - 장점: 이미지의 공간적 구조 보존, 특징 자동 추출  

- [RNN/LSTM](lectDL_NN/공기_RNN_LSTM.ipynb) - 시계열 데이터 예측  
**RNN/LSTM**: 순차 데이터 처리를 위한 순환 신경망  
    - RNN: 이전 시점의 정보를 현재 계산에 활용하는 순환 구조  
    - LSTM: 장기 의존성 문제를 해결한 개선된 RNN 구조  
    - 용도: 자연어처리, 시계열 예측, 음성 인식 등  

- [앙상블 모델](lectDL_NN/심장병_앙상블_비교.ipynb) - 다중 모델 성능 비교  
여러 딥러닝 모델의 앙상블 기법과 성능 평가


### Machine Learning

- [KNN 분류/회귀](lectMLDL/KNN_온도_예측.ipynb) - K-최근접 이웃 알고리즘  
**KNN**: 새로운 데이터를 분류할 때 가장 가까운 K개 이웃들의 다수결로 결정하는 알고리즘  
    - 거리 기반 학습: 유클리드 거리, 맨하탄 거리 등 사용  
    - 장점: 단순하고 직관적, 복잡한 결정경계 처리 가능  
    - 단점: 계산 비용 높음, 차원의 저주에 민감  

- [선형회귀](lectMLDL/다항선형회귀.ipynb) - 다항식 회귀 분석  
**Linear Regression**: 입력 변수와 출력 변수 사이의 선형 관계를 모델링하는 기법  
    - 목표: 최적의 직선(또는 평면)을 찾아 예측 수행  
    - 손실함수: MSE(평균제곱오차) 최소화  
    - 다항식 회귀: 비선형 관계를 다항식으로 확장  

- [결정트리](lectMLDL/DecisionTree.ipynb) - 의사결정트리 분류  
**Decision Tree**: 데이터를 여러 조건으로 분할하여 트리 구조로 분류/예측하는 알고리즘  
    - 분할 기준: 지니계수, 엔트로피, 정보이득 활용  
    - 장점: 해석하기 쉬움, 비선형 관계 처리 가능  
    - 단점: 과적합 경향, 데이터 변화에 민감  

- [전처리](lectMLDL/LabelEncoder.ipynb) - 데이터 전처리 기법  
라벨인코딩, 표준화, 정규화 등 데이터 변환


### Vision

- [YOLO 객체탐지](lectVision/yolo_1.ipynb) - 실시간 객체 인식  
**YOLO (You Only Look Once)**: 실시간 객체 탐지를 위한 원-스테이지 딥러닝 모델  
    - 특징: 이미지를 한 번만 보고 객체의 위치와 클래스를 동시에 예측  
    - 장점: 빠른 추론 속도, 실시간 처리 가능  
    - 구조: 입력 이미지를 그리드로 나누어 각 셀에서 바운딩박스와 클래스 확률 예측  

- [교통신호등 인식](lectVision/yolo_6_신호등.ipynb) - 특화 모델 구축  
YOLO 커스터마이징으로 교통신호등 인식 시스템

- [YOLO 튜토리얼](lectVision/yolo_tutorial.ipynb) - YOLO 기초 학습  
YOLO 모델 구조와 사용법, 성능 평가

### Crawling

- [네이버 뉴스](lectCrawling/네이버뉴스.ipynb) - 뉴스 데이터 수집  
BeautifulSoup으로 네이버 뉴스 크롤링, 워드클라우드 생성

- [쇼핑몰 분석](lectCrawling/쇼핑몰_분석.ipynb) - 전자상거래 데이터  
쇼핑몰 상품 정보 수집 및 시장 분석

- [IT 기사 수집](lectCrawling/네이버_IT_기사.ipynb) - 특정 카테고리 크롤링  
IT 관련 기사 자동 수집 및 트렌드 분석

### Pytorch

- [PyTorch 기초](lectPytorch/파이토치_타이타닉.ipynb) - 타이타닉 생존 예측  
PyTorch 기본 문법, 데이터로더, 모델 구축 기초

- [CNN 구현](lectPytorch/파이토치_CNN.ipynb) - 합성곱 신경망 구현  
PyTorch로 CNN 직접 구현, 이미지 분류 실습

- [시계열 예측](lectPytorch/파이토치_시계열.ipynb) - 시간 시리즈 모델링  
PyTorch 기반 시계열 데이터 예측 모델 구축

- [텍스트 처리](lectPytorch/파이토치_토큰화.ipynb) - 자연어 전처리  
토크나이징, 임베딩, 텍스트 데이터 처리 기법


## 🛠️ 기술 스택
- **언어**: Python
- **ML/DL**: PyTorch, Transformers, Scikit-learn
- **데이터**: NumPy, Pandas, Matplotlib, Seaborn
- **웹**: Streamlit, BeautifulSoup, Selenium
- **NLP**: Hugging Face, KoNLPy, WordCloud