#!/usr/bin/env python3
"""
CollegeQASystem - 구축된 벡터 스토어를 사용한 질문답변 전담 클래스

Author: kwangsiklee  
Version: 0.1.0
"""

import json
from pathlib import Path
from typing import Dict, Any

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class CollegeQASystem:
    """구축된 벡터 스토어를 사용한 질문답변 전담 클래스"""
    
    def __init__(self, vector_db_dir: str):
        self.vector_db_dir = Path(vector_db_dir)
        
        # LLM 설정 (필요시 지연 로딩)
        self.llm = None
        self.embeddings = None
        
        # 벡터 스토어와 QA 체인
        self.vector_store = None
        self.qa_chain = None
        
        # 프롬프트 템플릿
        self.setup_prompt_template()
        
        print(f"CollegeQASystem 초기화 완료")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
    
    def initialize_llm_components(self):
        """LLM 구성요소 지연 초기화"""
        if self.llm is None:
            print("🤖 LLM 구성요소 초기화 중...")
            
            # OpenAI 설정
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=800
            )
            
            # 임베딩 모델
            self.embeddings = OpenAIEmbeddings()
    
    def setup_prompt_template(self):
        """프롬프트 템플릿 설정"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""당신은 고등학생들의 대학교 전공 선택을 도와주는 전문 상담사입니다.

아래 대학교 학과 안내 자료를 바탕으로 학생의 질문에 정확하고 도움이 되는 답변을 해주세요.

참고 자료:
{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:
1. 고등학생이 이해하기 쉬운 언어로 설명해주세요
2. 구체적이고 실용적인 정보를 제공해주세요  
3. 진로와 관련된 조언을 포함해주세요
4. 참고 자료에 없는 내용은 일반적인 정보로 보완해주세요
5. 친근하고 격려하는 톤으로 답변해주세요
6. 참고 자료에 있는 답변과 없는 답변을 구분해서 표현해주세요
답변:"""
        )
    
    def load_vector_store(self):
        """기존 벡터 스토어 로드"""
        try:
            print("기존 벡터 스토어 로드 중...")
            
            # LLM 구성요소 초기화
            self.initialize_llm_components()
            
            self.vector_store = FAISS.load_local(
                str(self.vector_db_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # QA 체인 설정
            self.setup_qa_chain()
            
            # 메타데이터 로드
            metadata_path = self.vector_db_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✅ 벡터 스토어 로드 완료!")
                print(f"   생성일: {metadata.get('created_at', 'Unknown')}")
                print(f"   문서 수: {metadata.get('total_documents', 'Unknown')}")
            else:
                print("✅ 벡터 스토어 로드 완료!")
            
        except Exception as e:
            print(f"❌ 벡터 스토어 로드 실패: {e}")
            raise
    
    def setup_qa_chain(self):
        """QA 체인 설정"""
        if self.vector_store is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        # 리트리버 설정 (상위 3개 문서 검색)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template
            },
            return_source_documents=True
        )
        
        print("✅ QA 체인 설정 완료")
    
    def query(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        if self.qa_chain is None:
            raise ValueError("QA 체인이 설정되지 않았습니다. 먼저 벡터 스토어를 로드하세요.")
        
        try:
            print(f"🔍 질문 처리: {question}")
            
            # QA 체인으로 질문 처리
            result = self.qa_chain.invoke({"query": question})
            
            # 답변 추출
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            source_docs = result.get("source_documents", [])
            
            # 소스 정보 생성
            sources = []
            for doc in source_docs:
                metadata = doc.metadata
                source_info = f"{metadata.get('source', 'Unknown')} (페이지 {metadata.get('page', '?')})"
                if source_info not in sources:
                    sources.append(source_info)
            
            print(f"✅ 답변 생성 완료 (참고 자료: {len(sources)}개)")
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_documents": source_docs
            }
            
        except Exception as e:
            error_msg = f"질문 처리 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            
            return {
                "question": question,
                "answer": f"죄송합니다. {error_msg}",
                "sources": [],
                "source_documents": []
            }