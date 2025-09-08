#!/usr/bin/env python3
"""
VectorStoreBuilder - PDF 처리 및 벡터 스토어 구축 전담 클래스

Author: kwangsiklee  
Version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
from datetime import datetime
import gc

# LangChain imports
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# a_my_rag_module 추가
sys.path.append(str(Path(__file__).parent.parent))
from a_my_rag_module import PDFImageExtractor, KoreanOCR


class VectorStoreBuilder:
    """PDF 처리 및 벡터 스토어 구축 전담 클래스"""
    
    def __init__(self, pdf_dir: str, temp_images_dir: str, vector_db_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.temp_images_dir = Path(temp_images_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # 디렉토리 생성
        self.temp_images_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # PDF 처리 구성 요소 초기화
        self.pdf_extractor = PDFImageExtractor(dpi=100, max_size=2048)
        self.ocr = KoreanOCR()
        
        # 임베딩 모델 (필요시 지연 로딩)
        self.embeddings = None

        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 벡터 스토어 (구축 시에만 사용)
        self.vector_store = None
        
        print(f"VectorStoreBuilder 초기화 완료")
        print(f"PDF 디렉토리: {self.pdf_dir}")
        print(f"벡터 DB 디렉토리: {self.vector_db_dir}")
        
    def force_memory_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        
        # GPU 메모리 정리 (PyTorch)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def initialize_embedding_model(self):
        """임베딩 모델 지연 초기화"""
        if self.embeddings is None:
            print("🔤 임베딩 모델 초기화 중...")
            # 임베딩 모델
            self.embeddings = OpenAIEmbeddings()
    
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
            # sample_files = pdf_files[:1]
            sample_files = [self.pdf_dir / "01-경영대학.pdf"]
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
                    
                    check_memory = True # = self.check_memory_threshold()
                    if check_memory:
                        print("⚠️ 메모리 임계값 초과 - 강제 정리 및 대기")
                        self.force_memory_cleanup()
                        import time
                        time.sleep(2)  # 메모리 안정화 대기
                    
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
            
            # memory release ocr model
            self.ocr.cleanup_ocr_model()

            if not all_documents:
                raise ValueError("처리된 문서가 없습니다.")
            
            print(f"\n📊 총 문서 수: {len(all_documents)}개")
            
            if progress_callback:
                progress_callback(f"벡터 임베딩 생성 중... ({len(all_documents)}개 문서)")
            
            # 4. 임베딩 모델 초기화
            self.initialize_embedding_model()

            # 문서를 작은 배치로 나누어 처리 (메모리 절약)
            batch_size = 50  # 한 번에 50개 문서씩
            
            if len(all_documents) <= batch_size:
                # 문서가 적으면 한 번에 처리
                self.vector_store = FAISS.from_documents(
                    documents=all_documents,
                    embedding=self.embeddings
                )
            else:
                # 배치로 나누어 처리
                print(f"   📦 배치 처리: {batch_size}개씩")
                
                # 첫 번째 배치로 벡터 스토어 생성
                first_batch = all_documents[:batch_size]
                self.vector_store = FAISS.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings
                )
                
                # 나머지 배치들을 추가
                for i in range(batch_size, len(all_documents), batch_size):
                    batch = all_documents[i:i + batch_size]
                    batch_vector_store = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                    
                    # 벡터 스토어 병합
                    self.vector_store.merge_from(batch_vector_store)
                    
                    # 배치 처리 후 메모리 정리
                    batch_vector_store = None
                    gc.collect()
                    
                    print(f"   ✅ 배치 {i//batch_size + 1} 완료")

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
    
    def initialize_vector_db(self, force_rebuild: bool = False, progress_callback: Optional[Callable] = None):
        """벡터 DB 초기화 - 독립 실행 가능한 초기화 함수"""
        try:
            if progress_callback:
                progress_callback("벡터 DB 초기화 시작...")
            
            print("🔄 벡터 DB 초기화 시작...")
            
            # 기존 벡터 DB 확인
            if self.vector_store_exists() and not force_rebuild:
                if progress_callback:
                    progress_callback("기존 벡터 DB 발견 - 검증 중...")
                
                print("📁 기존 벡터 DB 발견 - 검증 시도...")
                try:
                    self.initialize_embedding_model()
                    # 벡터 스토어 존재 확인만 수행
                    _ = FAISS.load_local(
                        str(self.vector_db_dir),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    if progress_callback:
                        progress_callback("✅ 기존 벡터 DB 검증 완료!")
                    
                    print("✅ 기존 벡터 DB 검증 완료!")
                    return True, "기존 벡터 DB 검증 완료."
                
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"기존 DB 검증 실패 - 새로 구축: {e}")
                    
                    print(f"⚠️ 기존 DB 검증 실패 - 새로 구축합니다: {e}")
                    force_rebuild = True
            
            # 새 벡터 DB 구축 또는 강제 재구축
            if not self.vector_store_exists() or force_rebuild:
                if force_rebuild:
                    if progress_callback:
                        progress_callback("기존 벡터 DB 삭제 후 새로 구축...")
                    
                    print("🗑️ 기존 벡터 DB 삭제 후 새로 구축...")
                    # 기존 벡터 DB 파일 삭제
                    import shutil
                    if self.vector_db_dir.exists():
                        shutil.rmtree(self.vector_db_dir)
                        self.vector_db_dir.mkdir(exist_ok=True)
                
                if progress_callback:
                    progress_callback("새 벡터 DB 구축 시작...")
                
                print("🏗️ 새 벡터 DB 구축 시작...")
                self.build_vector_store(progress_callback)
                
                if progress_callback:
                    progress_callback("✅ 새 벡터 DB 구축 완료!")
                
                print("✅ 새 벡터 DB 구축 완료!")
                return True, "새 벡터 DB를 성공적으로 구축했습니다."
            
        except Exception as e:
            error_msg = f"벡터 DB 초기화 실패: {e}"
            print(f"❌ {error_msg}")
            
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
            
            return False, error_msg
    
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