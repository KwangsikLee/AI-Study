"""
FAISS 벡터 스토어 저장/로드 사용 예제
"""

from embedding import VectorStoreManager
from langchain.schema import Document

def example_usage():
    """FAISS 저장/로드 기능 사용 예제"""
    
    # 1. VectorStoreManager 초기화
    print("=== 1. VectorStoreManager 초기화 ===")
    manager = VectorStoreManager(
        embedding_model_key="ko-sroberta-multitask",
        save_directory="./faiss_indexes"  # 저장할 디렉토리 지정
    )
    
    # 2. 샘플 문서 생성
    print("\n=== 2. 샘플 문서 생성 ===")
    sample_docs = [
        Document(
            page_content="LangChain은 대화형 AI 애플리케이션을 구축하기 위한 프레임워크입니다.",
            metadata={"source": "langchain_intro", "topic": "AI"}
        ),
        Document(
            page_content="FAISS는 Facebook AI에서 개발한 고성능 벡터 유사도 검색 라이브러리입니다.",
            metadata={"source": "faiss_intro", "topic": "Vector Search"}
        ),
        Document(
            page_content="RAG는 검색 증강 생성으로, 외부 지식을 활용한 텍스트 생성 방법입니다.",
            metadata={"source": "rag_intro", "topic": "Generation"}
        ),
        Document(
            page_content="임베딩은 텍스트를 수치 벡터로 변환하는 과정입니다.",
            metadata={"source": "embedding_intro", "topic": "NLP"}
        )
    ]
    
    print(f"생성된 문서 수: {len(sample_docs)}")
    
    # 3. 벡터 스토어 생성 및 자동 저장
    print("\n=== 3. 벡터 스토어 생성 및 자동 저장 ===")
    vector_store = manager.auto_save_after_creation(
        documents=sample_docs,
        index_name="sample_docs",
        model_key="ko-sroberta-multitask"
    )
    
    # 4. 검색 테스트
    print("\n=== 4. 검색 테스트 ===")
    query = "벡터 검색이란 무엇인가요?"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"검색 쿼리: {query}")
    print("검색 결과:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.page_content[:50]}...")
        print(f"   출처: {result.metadata.get('source', 'Unknown')}")
    
    # 5. 저장된 인덱스 목록 조회
    print("\n=== 5. 저장된 인덱스 목록 조회 ===")
    saved_indexes = manager.list_saved_indexes()
    print("저장된 인덱스:")
    for index_name, models in saved_indexes.items():
        print(f"📁 {index_name}")
        for model_key, info in models.items():
            print(f"  └── {model_key}: {info['document_count']}개 문서 ({info['created_at'][:16]})")
    
    # 6. 새로운 매니저로 인덱스 로드 테스트
    print("\n=== 6. 새로운 매니저로 인덱스 로드 테스트 ===")
    new_manager = VectorStoreManager(save_directory="./faiss_indexes")
    
    # 저장된 인덱스 로드
    load_result = new_manager.load_vector_store(
        index_name="sample_docs",
        model_key="ko-sroberta-multitask"
    )
    print(load_result)
    
    # 로드된 벡터 스토어로 검색 테스트
    if new_manager.current_vector_store:
        print("\n로드된 벡터 스토어로 검색:")
        query2 = "LangChain이란?"
        results2 = new_manager.current_vector_store.similarity_search(query2, k=1)
        for result in results2:
            print(f"결과: {result.page_content}")
    
    # 7. 다른 임베딩 모델로 벡터 스토어 생성
    print("\n=== 7. 다른 임베딩 모델로 벡터 스토어 생성 ===")
    manager.auto_save_after_creation(
        documents=sample_docs[:2],  # 일부 문서만 사용
        index_name="sample_docs",
        model_key="paraphrase-multilingual"
    )
    
    # 8. 최종 저장된 인덱스 목록
    print("\n=== 8. 최종 저장된 인덱스 목록 ===")
    final_indexes = manager.list_saved_indexes()
    for index_name, models in final_indexes.items():
        print(f"📁 {index_name}")
        for model_key, info in models.items():
            print(f"  └── {model_key}: {info['document_count']}개 문서")
    
    print("\n✅ 모든 테스트가 완료되었습니다!")
    print("저장된 파일들은 './faiss_indexes' 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    example_usage()