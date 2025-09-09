from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.retriever import BaseRetriever
from pydantic import Field, ConfigDict




# =============================================================================
#  Reranker 클래스 (재순위화)
# =============================================================================

class MyReranker:
    def __init__(self):
        # 사용 가능한 reranker 모델들
        self.reranker_models = {
            "cross-encoder-ms-marco": {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "description": "MS MARCO 데이터셋으로 학습된 경량 CrossEncoder",
                "language": "English (Multilingual capable)",
                "size": "~80MB"
            },
            "cross-encoder-multilingual": {
                "name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "description": "다국어 mMARCO CrossEncoder",
                "language": "Multilingual",
                "size": "~400MB"
            },
            "bge-reranker": {
                "name": "BAAI/bge-reranker-base",
                "description": "BGE Reranker (중국어/영어 특화)",
                "language": "Chinese/English",
                "size": "~400MB"
            }
        }

        self.loaded_rerankers = {}
        self.current_reranker = None
        self.current_model_key = None

    def load_reranker(self, model_key: str = "cross-encoder-ms-marco") -> CrossEncoder:
        """Reranker 모델 로드"""
        if model_key in self.loaded_rerankers:
            self.current_reranker = self.loaded_rerankers[model_key]
            self.current_model_key = model_key
            return self.current_reranker

        if model_key not in self.reranker_models:
            model_key = "cross-encoder-ms-marco"  # fallback

        model_info = self.reranker_models[model_key]
        print(f"Loading reranker: {model_info['name']} ({model_info['size']})")

        try:
            reranker = CrossEncoder(model_info['name'])
            self.loaded_rerankers[model_key] = reranker
            self.current_reranker = reranker
            self.current_model_key = model_key
            return reranker
        except Exception as e:
            print(f"Failed to load reranker {model_key}: {e}")
            # fallback to None (no reranking)
            self.current_reranker = None
            return None

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """문서 재순위화"""
        if not self.current_reranker or len(documents) <= 1:
            return documents[:top_k]

        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [(query, doc.page_content[:512]) for doc in documents]

            # 점수 계산
            scores = self.current_reranker.predict(query_doc_pairs)

            # 점수와 문서를 결합하여 정렬
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # 상위 top_k 개 문서 반환
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]

            print(f"Reranked {len(documents)} documents to top {len(reranked_docs)} with model: {self.current_model_key}")
            return reranked_docs

        except Exception as e:
            print(f"Reranking failed: {e}, returning original order")
            return documents[:top_k]

    def get_available_rerankers(self) -> Dict[str, str]:
        """사용 가능한 reranker 모델 목록"""
        models = {key: f"{info['description']} ({info['language']}, {info['size']})"
                 for key, info in self.reranker_models.items()}
        models["none"] = "Reranking 사용하지 않음"
        return models

    def get_current_reranker_info(self) -> str:
        """현재 reranker 정보"""
        if self.current_reranker and self.current_model_key:
            info = self.reranker_models[self.current_model_key]
            return f"🔄 {info['description']} ({info['language']}, {info['size']})"
        return "🔄 Reranking 사용하지 않음"



# =============================================================================
# 향상된 복합 검색 시스템 클래스 (Retrieval - 복수의 모델 적용 + Rerank)
# =============================================================================

class AdvancedHybridRetriever:
    def __init__(self, documents: List[Document] = None, vector_store: FAISS = None,
                 reranker: MyReranker = None, reranker_model: str = "cross-encoder-ms-marco"):
        self.vector_store = vector_store
        self.reranker = reranker or MyReranker()

        # Reranker 로드
        if reranker_model != "none":
            self.reranker.load_reranker(reranker_model)

        # 문서 추출: documents가 없으면 vector_store에서 추출
        if documents is not None:
            self.documents = documents
        elif vector_store is not None:
            self.documents = self._extract_documents_from_vector_store(vector_store)
        else:
            raise ValueError("documents 또는 vector_store 중 하나는 반드시 제공되어야 합니다.")

        # BM25 검색기 생성
        self._create_bm25_retriever()

        # 벡터 검색기
        self.vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 15}  # rerank를 위해 더 많이 가져옴
        )

        # 앙상블 검색기 (하이브리드)
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[0.4, 0.6]
            )
        else:
            self.ensemble_retriever = self.vector_retriever

    def _extract_documents_from_vector_store(self, vector_store: FAISS) -> List[Document]:
        """FAISS 벡터 스토어에서 문서들을 추출"""
        documents = []
        try:
            # FAISS 벡터 스토어의 docstore에서 문서 추출
            if hasattr(vector_store, 'docstore') and hasattr(vector_store, 'index_to_docstore_id'):
                print(f"벡터 스토어에서 {len(vector_store.index_to_docstore_id)}개 문서 추출 중...")
                
                for i, doc_id in enumerate(vector_store.index_to_docstore_id.values()):
                    try:
                        doc = vector_store.docstore.search(doc_id)
                        if doc:
                            documents.append(doc)
                    except Exception as e:
                        print(f"문서 {i} 추출 실패: {e}")
                        continue
                        
                print(f"✅ 벡터 스토어에서 {len(documents)}개 문서 추출 완료")
                
            else:
                print("⚠️ 벡터 스토어에서 docstore를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 벡터 스토어에서 문서 추출 실패: {e}")
            
        return documents
    
    def _create_bm25_retriever(self):
        """BM25 검색기 생성"""
        try:
            if not self.documents:
                print("⚠️ 문서가 없어서 BM25 검색기를 생성할 수 없습니다.")
                self.bm25_retriever = None
                return
                
            texts = [doc.page_content for doc in self.documents]
            metadatas = [doc.metadata for doc in self.documents]
            
            self.bm25_retriever = BM25Retriever.from_texts(
                texts,
                metadatas=metadatas
            )
            self.bm25_retriever.k = 15  # rerank를 위해 더 많이 가져옴
            print(f"✅ BM25 검색기 생성 완료 ({len(texts)}개 문서)")
            
        except Exception as e:
            print(f"❌ BM25 검색기 생성 오류: {e}")
            self.bm25_retriever = None

    def search_by_similarity(self, query: str, k: int = 5, use_rerank: bool = True) -> List[Document]:
        """유사도 기반 검색 + Rerank"""
        docs = self.vector_retriever.invoke(query)
        if use_rerank and self.reranker.current_reranker:
            docs = self.reranker.rerank_documents(query, docs, k)
        return docs[:k]

    def search_by_keyword(self, query: str, k: int = 5, use_rerank: bool = True) -> List[Document]:
        """키워드 기반 검색 + Rerank"""
        if self.bm25_retriever:
            docs = self.bm25_retriever.invoke(query)
        else:
            docs = self.vector_retriever.invoke(query)

        if use_rerank and self.reranker.current_reranker:
            docs = self.reranker.rerank_documents(query, docs, k)
        return docs[:k]

    def hybrid_search(self, query: str, k: int = 5, use_rerank: bool = True) -> List[Document]:
        """하이브리드 검색 + Rerank"""
        docs = self.ensemble_retriever.invoke(query)
        if use_rerank and self.reranker.current_reranker:
            docs = self.reranker.rerank_documents(query, docs, k)
        return docs[:k]

    def advanced_search(self, query: str, method: str = "hybrid", k: int = 5,
                       use_rerank: bool = True, rerank_top_k: int = 15) -> List[Document]:
        """고급 검색 (모든 방법 + Rerank)"""
        if method == "similarity":
            # 벡터 검색
            docs = self.vector_retriever.invoke(query)[:rerank_top_k]
        elif method == "keyword":
            # BM25 검색
            if self.bm25_retriever:
                docs = self.bm25_retriever.invoke(query)[:rerank_top_k]
            else:
                docs = self.vector_retriever.invoke(query)[:rerank_top_k]
        elif method == "ensemble":
            # 앙상블 검색 (BM25 + Vector)
            docs = self.ensemble_retriever.invoke(query)[:rerank_top_k]
        else:  # method == "hybrid"
            # 다중 방법 융합
            vector_docs = self.vector_retriever.invoke(query)[:10]
            if self.bm25_retriever:
                keyword_docs = self.bm25_retriever.invoke(query)[:10]
                # 문서 융합 (중복 제거)
                seen_docs = set()
                fused_docs = []
                for doc in vector_docs + keyword_docs:
                    doc_id = doc.page_content[:100]  # 간단한 중복 체크
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        fused_docs.append(doc)
                docs = fused_docs[:rerank_top_k]
            else:
                docs = vector_docs

        # Rerank 적용
        if use_rerank and self.reranker.current_reranker:
            docs = self.reranker.rerank_documents(query, docs, k)

        return docs[:k]

    def search_by_date(self, date: str) -> List[Document]:
        """날짜 기반 검색"""
        filtered_docs = [doc for doc in self.documents
                        if doc.metadata.get('date') == date]
        return filtered_docs

    def switch_reranker(self, reranker_model: str) -> str:
        """Reranker 모델 변경"""
        if reranker_model == "none":
            self.reranker.current_reranker = None
            return "🔄 Reranking이 비활성화되었습니다."
        else:
            self.reranker.load_reranker(reranker_model)
            return f"🔄 Reranker 변경 완료: {self.reranker.get_current_reranker_info()}"


# =============================================================================
# LangChain 호환 Wrapper 클래스
# =============================================================================

class HybridRetrieverWrapper(BaseRetriever):
    """AdvancedHybridRetriever를 내부에서 생성하여 LangChain과 호환되도록 하는 래퍼 클래스"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    vector_store: Any = Field(description="FAISS vector store instance")
    reranker_model: str = Field(default="cross-encoder-ms-marco", description="Reranker model name")
    
    def __init__(self, vector_store: FAISS, reranker_model: str = "cross-encoder-ms-marco", **kwargs):
        super().__init__(
            vector_store=vector_store, 
            reranker_model=reranker_model, 
            **kwargs
        )
        
        # AdvancedHybridRetriever를 일반 인스턴스 속성으로 생성
        print("🔧 HybridRetrieverWrapper에서 AdvancedHybridRetriever 생성 중...")
        reranker = MyReranker()
        self.hybrid_retriever = AdvancedHybridRetriever(
            vector_store=vector_store,
            reranker=reranker,
            reranker_model=reranker_model
        )
        print("✅ HybridRetrieverWrapper 초기화 완료")
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """LangChain BaseRetriever의 필수 메소드 구현"""
        # hybrid_search 메소드 사용 (상위 3개 문서 검색, reranking 적용)
        return self.hybrid_retriever.hybrid_search(
            query=query, 
            k=3, 
            use_rerank=True
        )
    
    def switch_reranker(self, reranker_model: str) -> str:
        """Reranker 모델 변경"""
        if self.hybrid_retriever:
            return self.hybrid_retriever.switch_reranker(reranker_model)
        return "❌ HybridRetriever가 초기화되지 않았습니다."
    
    def get_retriever_info(self) -> str:
        """현재 검색기 정보 반환"""
        if self.hybrid_retriever:
            info = ["📊 HybridRetrieverWrapper 정보"]
            info.append(f"   문서 수: {len(self.hybrid_retriever.documents) if self.hybrid_retriever.documents else 0}개")
            info.append(f"   BM25 검색기: {'활성화' if self.hybrid_retriever.bm25_retriever else '비활성화'}")
            info.append(f"   벡터 검색기: 활성화")
            if self.hybrid_retriever.reranker:
                info.append(f"   {self.hybrid_retriever.reranker.get_current_reranker_info()}")
            return "\n".join(info)
        return "❌ HybridRetriever가 초기화되지 않았습니다."


if __name__ == "__main__":
    import sys
    import os
    # 현재 파일의 부모 디렉토리를 sys.path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from a_my_rag_module.embedding import VectorStoreManager

    # 1. Documents
    documents = [
        Document(
            page_content="""
            당뇨병은 인슐린의 분비량이 부족하거나 정상적인 기능이 이루어지지 않는 대사질환입니다.
            제1형 당뇨병은 인슐린을 생산하지 못해 발생하며, 주로 소아청소년기에 발병합니다.
            제2형 당뇨병은 인슐린 저항성으로 인해 발생하며, 성인에게서 주로 나타납니다.
            당뇨병의 주요 증상으로는 다뇨, 다음, 다식, 체중감소 등이 있습니다.
            혈당 관리를 위해서는 규칙적인 운동과 식이조절이 필수적입니다.
            """,
            metadata={"category": "disease", "topic": "diabetes", "doc_id": "1"}
        ),
        Document(
            page_content="""
            고혈압은 수축기 혈압이 140mmHg 이상이거나 이완기 혈압이 90mmHg 이상인 상태를 말합니다.
            고혈압은 '침묵의 살인자'로 불리며, 초기에는 특별한 증상이 없는 경우가 많습니다.
            장기간 고혈압이 지속되면 심장질환, 뇌졸중, 신장질환의 위험이 증가합니다.
            생활습관 개선과 약물치료를 통해 혈압을 관리할 수 있습니다.
            정기적인 혈압 측정과 모니터링이 중요합니다.
            """,
            metadata={"category": "disease", "topic": "hypertension", "doc_id": "2"}
        )
    ]

    # 3. 임베딩 모델 로드 및 벡터 스토어 생성
    print("3. 임베딩 모델 로드 및 벡터 스토어 생성 중...")
    embedding_model_key: str = "ko-sroberta-multitask"
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.create_vector_store(documents)

    # 5. 고급 검색기 생성
    print("5. 고급 하이브리드 검색기 생성 중...")
    retriever = AdvancedHybridRetriever(documents, vector_store)
