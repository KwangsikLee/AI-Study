#!/usr/bin/env python3
"""
대학 학과 선택 도우미 RAG 시스템 구현
PDF 이미지 추출 → OCR → 벡터 임베딩 → 검색 → LLM 답변 생성

Author: kwangsiklee  
Version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pickle
import json
from datetime import datetime

# LangChain imports
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# a_my_rag_module 추가
sys.path.append(str(Path(__file__).parent.parent))
from a_my_rag_module import PDFImageExtractor, KoreanOCR, MultiEmbeddingManager


class CollegeRAGSystem:
    """대학교 학과 안내 자료 기반 RAG 시스템"""
    
    def __init__(self, pdf_dir: str, temp_images_dir: str, vector_db_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.temp_images_dir = Path(temp_images_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # 디렉토리 생성
        self.temp_images_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # 구성 요소 초기화
        self.pdf_extractor = PDFImageExtractor(dpi=150, max_size=2048)
        self.ocr = KoreanOCR()
        
        # OpenAI 설정
        self.llm = ChatOpenAI(
            model= "gpt-4o-mini",  #"gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000
        )
        
        # 임베딩 모델
        self.embeddings = OpenAIEmbeddings()
        
        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 벡터 스토어
        self.vector_store = None
        self.qa_chain = None
        
        # 프롬프트 템플릿
        self.setup_prompt_template()
        
        print(f"CollegeRAGSystem 초기화 완료")
        print(f"PDF 디렉토리: {self.pdf_dir}")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
    
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

답변:"""
        )
    
    def vector_store_exists(self) -> bool:
        """벡터 스토어가 이미 존재하는지 확인"""
        faiss_index_path = self.vector_db_dir / "index.faiss"
        faiss_pkl_path = self.vector_db_dir / "index.pkl"
        return faiss_index_path.exists() and faiss_pkl_path.exists()
    
    def build_vector_store(self, progress_callback: Optional[Callable] = None):
        """PDF 파일들을 처리하여 벡터 스토어 구축"""
        try:
            if progress_callback:
                progress_callback("PDF 파일 목록 확인 중...")
            
            # PDF 파일 목록 가져오기
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            
            if not pdf_files:
                raise ValueError(f"PDF 파일이 없습니다: {self.pdf_dir}")
            
            print(f"발견된 PDF 파일: {len(pdf_files)}개")
            
            all_documents = []
            
            # 샘플로 처음 1개 파일만 처리 (MVP)
            sample_files = pdf_files[:1]
            
            for i, pdf_file in enumerate(sample_files):
                try:
                    if progress_callback:
                        progress_callback(f"처리 중: {pdf_file.name} ({i+1}/{len(sample_files)})")
                    
                    print(f"\n📄 처리 중: {pdf_file.name}")
                    
                    # 1. PDF에서 이미지 추출
                    image_paths = self.pdf_extractor.extract_images_from_pdf(
                        str(pdf_file),
                        str(self.temp_images_dir),
                        split_large_pages=True
                    )
                    
                    print(f"  📷 추출된 이미지: {len(image_paths)}개")
                    
                    # 2. OCR로 텍스트 추출
                    pdf_texts = []
                    for img_path in image_paths:
                        try:
                            text = self.ocr.extract_text(img_path)
                            if text.strip():  # 빈 텍스트가 아닌 경우만
                                pdf_texts.append(text.strip())
                        except Exception as e:
                            print(f"    ⚠️ OCR 실패: {img_path} - {e}")
                            continue
                    
                    print(f"  📝 추출된 텍스트 블록: {len(pdf_texts)}개")
                    
                    # 3. 텍스트를 Document 객체로 변환
                    for j, text in enumerate(pdf_texts):
                        if len(text) > 50:  # 너무 짧은 텍스트 제외
                            # 텍스트 분할
                            chunks = self.text_splitter.split_text(text)
                            
                            for k, chunk in enumerate(chunks):
                                doc = Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": pdf_file.name,
                                        "page": j + 1,
                                        "chunk": k + 1,
                                        "processed_at": datetime.now().isoformat()
                                    }
                                )
                                all_documents.append(doc)
                    
                    print(f"  ✅ 완료: {pdf_file.name}")
                    
                except Exception as e:
                    print(f"  ❌ 오류: {pdf_file.name} - {e}")
                    continue
            
            if not all_documents:
                raise ValueError("처리된 문서가 없습니다.")
            
            print(f"\n📊 총 문서 수: {len(all_documents)}개")
            
            if progress_callback:
                progress_callback(f"벡터 임베딩 생성 중... ({len(all_documents)}개 문서)")
            
            # 4. 벡터 스토어 생성
            self.vector_store = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings
            )
            
            # 5. 벡터 스토어 저장
            self.vector_store.save_local(str(self.vector_db_dir))
            
            # 6. 메타데이터 저장
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_documents": len(all_documents),
                "processed_pdfs": [f.name for f in sample_files],
                "vector_db_path": str(self.vector_db_dir)
            }
            
            metadata_path = self.vector_db_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 7. QA 체인 설정
            self.setup_qa_chain()
            
            if progress_callback:
                progress_callback(f"벡터 스토어 구축 완료! (문서 {len(all_documents)}개)")
            
            print(f"✅ 벡터 스토어 구축 완료!")
            print(f"   저장 위치: {self.vector_db_dir}")
            print(f"   총 문서: {len(all_documents)}개")
            
        except Exception as e:
            error_msg = f"벡터 스토어 구축 실패: {e}"
            print(f"❌ {error_msg}")
            if progress_callback:
                progress_callback(error_msg)
            raise
    
    def load_vector_store(self):
        """기존 벡터 스토어 로드"""
        try:
            print("기존 벡터 스토어 로드 중...")
            
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
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """벡터 스토어 정보 반환"""
        info = {
            "exists": self.vector_store_exists(),
            "initialized": self.vector_store is not None,
            "vector_db_path": str(self.vector_db_dir)
        }
        
        # 메타데이터 정보 추가
        metadata_path = self.vector_db_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception as e:
                info["metadata_error"] = str(e)
        
        return info


# 테스트 함수
def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 RAG 시스템 테스트 시작")
    
    # 경로 설정
    base_dir = Path(__file__).parent
    pdf_dir = base_dir / "korea_univ_guides"
    temp_images_dir = base_dir / "temp_images"  
    vector_db_dir = base_dir / "vector_db"
    
    # RAG 시스템 초기화
    rag_system = CollegeRAGSystem(
        pdf_dir=str(pdf_dir),
        temp_images_dir=str(temp_images_dir),
        vector_db_dir=str(vector_db_dir)
    )
    
    # 벡터 스토어 구축 또는 로드
    if not rag_system.vector_store_exists():
        print("새로운 벡터 스토어 구축...")
        rag_system.build_vector_store()
    else:
        print("기존 벡터 스토어 로드...")
        rag_system.load_vector_store()
    
    # 테스트 질문들
    test_questions = [
        "컴퓨터공학과는 어떤 공부를 하나요?",
        "경영학과의 주요 과목은 무엇인가요?",
        "의대 진학을 위해 어떤 준비가 필요한가요?"
    ]
    
    print("\n🤖 질문-답변 테스트:")
    for question in test_questions:
        print(f"\n❓ {question}")
        result = rag_system.query(question)
        print(f"💬 {result['answer']}")
        if result['sources']:
            print(f"📚 참고 자료: {', '.join(result['sources'])}")


if __name__ == "__main__":
    # 환경 변수 로드
    from dotenv import load_dotenv
    load_dotenv()
    test_rag_system()