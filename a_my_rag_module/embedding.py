from typing import List, Dict, Any, Optional
import os
import pickle
# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from sentence_transformers import SentenceTransformer

# =============================================================================
# 다중 임베딩 및 벡터 DB 관리 클래스 (임베딩 & 벡터 DB화)
# =============================================================================

class MultiEmbeddingManager:
    def __init__(self, api_token):
        # 사용 가능한 임베딩 모델들
        self.embedding_models = {
            "embedding-gemma": {
                "name": "google/embeddinggemma-300m",
                "description": "다국어 지원 SentenceTransformer 모델",
                "language": "Multilingual",
                "size": "~1.2G"
            },
            "ko-sroberta-multitask": {
                "name": "jhgan/ko-sroberta-multitask",
                "description": "한국어 특화 SentenceTransformer 모델",
                "language": "Korean",
                "size": "~300MB"
            },
            "paraphrase-multilingual": {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "다국어 지원 경량 모델",
                "language": "Multilingual",
                "size": "~420MB"
            },
            "ko-sentence-transformer": {
                "name": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                "description": "100개 언어 지원 BERT 기반 모델",
                "language": "Multilingual",
                "size": "~1.1GB"
            },
            "distiluse-multilingual": {
                "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "description": "다국어 DistilUSE 모델 (빠름)",
                "language": "Multilingual",
                "size": "~540MB"
            },
            "ko-electra": {
                "name": "bongsoo/kpf-sbert-128d",
                "description": "한국어 ELECTRA 기반 경량 모델",
                "language": "Korean",
                "size": "~50MB"
            }
        }

        self.loaded_embeddings = {}
        self.current_model = None
        self.hf_api_token = api_token
        print(f"embedding manager with key: {self.hf_api_token}")

    def load_embedding_model(self, model_key: str) -> HuggingFaceEmbeddings:
        """임베딩 모델 로드"""
        if model_key in self.loaded_embeddings:
            return self.loaded_embeddings[model_key]

        if model_key not in self.embedding_models:
            raise ValueError(f"Unknown embedding model: {model_key}")

        model_info = self.embedding_models[model_key]
        print(f"Loading embedding model: {model_info['name']} ({model_info['size']})")
        print(f"my hf api token = {self.hf_api_token}")   
        try:

            if model_key == "embedding-gemma":
                
                # # Download from the 🤗 Hub
                # model = SentenceTransformer("google/embeddinggemma-300m")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_info['name'],
                    model_kwargs={'device': 'cpu', "token": self.hf_api_token}
                )
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_info['name'],
                    model_kwargs={'device': 'cpu', "token": self.hf_api_token}
                )
            self.loaded_embeddings[model_key] = embeddings
            self.current_model = model_key
            return embeddings
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")
            # fallback to default model
            if model_key != "paraphrase-multilingual":
                return self.load_embedding_model("paraphrase-multilingual")
            raise e

    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return {key: f"{info['description']} ({info['language']}, {info['size']})"
                for key, info in self.embedding_models.items()}

    def get_current_model_info(self) -> str:
        """현재 사용 중인 모델 정보 반환"""
        if self.current_model:
            info = self.embedding_models[self.current_model]
            return f"🤖 {info['description']} ({info['language']}, {info['size']})"
        return "모델이 로드되지 않음"

class VectorStoreManager:
    def __init__(self, embedding_model_key: str = "embedding-gemma", save_directory: str = "faiss_indexes", hf_api_token: str = None):
        self.embedding_manager = MultiEmbeddingManager(api_token=hf_api_token)
        self.embeddings = self.embedding_manager.load_embedding_model(embedding_model_key)
        self.vector_stores = {}  # 모델별 벡터 스토어 저장
        self.current_vector_store = None
        self.current_model_key = embedding_model_key
        self.save_directory = save_directory
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_directory, exist_ok=True)

    def switch_embedding_model(self, model_key: str) -> str:
        """임베딩 모델 변경"""
        try:
            print(f"Switching to embedding model: {model_key}")
            self.embeddings = self.embedding_manager.load_embedding_model(model_key)
            self.current_model_key = model_key

            # 기존 벡터 스토어가 있다면 사용, 없다면 새로 생성 필요
            if model_key in self.vector_stores:
                self.current_vector_store = self.vector_stores[model_key]
                return f"✅ 모델 변경 완료: {self.embedding_manager.get_current_model_info()}"
            else:
                return f"✅ 모델 변경 완료 (벡터 스토어 재생성 필요): {self.embedding_manager.get_current_model_info()}"
        except Exception as e:
            return f"❌ 모델 변경 실패: {str(e)}"

    def create_vector_store(self, documents: List[Document], model_key: str = None) -> FAISS:
        """벡터 스토어 생성"""
        if model_key and model_key != self.current_model_key:
            self.switch_embedding_model(model_key)

        print(f"임베딩 및 벡터 스토어 생성 중... (모델: {self.current_model_key})")

        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_stores[self.current_model_key] = vector_store
            self.current_vector_store = vector_store
            return vector_store
        except Exception as e:
            print(f"벡터 스토어 생성 실패: {e}")
            # fallback model로 재시도
            if self.current_model_key != "paraphrase-multilingual":
                print("기본 모델로 재시도...")
                self.switch_embedding_model("paraphrase-multilingual")
                vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_stores[self.current_model_key] = vector_store
                self.current_vector_store = vector_store
                return vector_store
            raise e

    def add_documents(self, documents: List[Document]):
        """기존 벡터 스토어에 문서 추가"""
        if self.current_vector_store:
            self.current_vector_store.add_documents(documents)
            self.vector_stores[self.current_model_key] = self.current_vector_store
        else:
            self.create_vector_store(documents)

    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 임베딩 모델 목록"""
        return self.embedding_manager.get_available_models()

    def get_current_model_info(self) -> str:
        """현재 모델 정보"""
        return self.embedding_manager.get_current_model_info()
    
    def save_vector_store(self, index_name: str, model_key: str = None) -> str:
        """벡터 스토어를 파일로 저장"""
        if model_key is None:
            model_key = self.current_model_key
            
        if model_key not in self.vector_stores:
            return f"❌ 저장할 벡터 스토어가 없습니다: {model_key}"
        
        try:
            # FAISS 인덱스 저장 경로
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            
            # FAISS 인덱스 저장
            vector_store = self.vector_stores[model_key]
            vector_store.save_local(index_path)
            
            # 메타데이터 저장 (모델 정보, 생성일시 등)
            metadata = {
                'model_key': model_key,
                'model_name': self.embedding_manager.embedding_models[model_key]['name'],
                'index_name': index_name,
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'document_count': len(vector_store.docstore._dict) if hasattr(vector_store.docstore, '_dict') else 'unknown'
            }
            
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            return f"✅ 벡터 스토어 저장 완료: {index_path}"
            
        except Exception as e:
            return f"❌ 벡터 스토어 저장 실패: {str(e)}"
    
    def load_vector_store(self, index_name: str, model_key: str = None) -> str:
        """저장된 벡터 스토어를 로드"""
        if model_key is None:
            model_key = self.current_model_key
        
        try:
            # FAISS 인덱스 로드 경로
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            
            if not os.path.exists(index_path):
                return f"❌ 벡터 스토어 파일이 없습니다: {index_path}"
            
            # 해당 모델의 임베딩이 로드되지 않았다면 로드
            if model_key != self.current_model_key:
                self.switch_embedding_model(model_key)
            
            # FAISS 인덱스 로드
            vector_store = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            self.vector_stores[model_key] = vector_store
            self.current_vector_store = vector_store
            
            # 메타데이터 로드 (있다면)
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            metadata_info = ""
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    metadata_info = f" (생성일: {metadata.get('created_at', 'Unknown')}, 문서수: {metadata.get('document_count', 'Unknown')})"
            
            return f"✅ 벡터 스토어 로드 완료: {index_name}_{model_key}{metadata_info}"
            
        except Exception as e:
            return f"❌ 벡터 스토어 로드 실패: {str(e)}"
    
    def list_saved_indexes(self) -> Dict[str, Dict[str, Any]]:
        """저장된 인덱스 목록 반환"""
        saved_indexes = {}
        
        if not os.path.exists(self.save_directory):
            return saved_indexes
        
        for filename in os.listdir(self.save_directory):
            if filename.endswith('.faiss'):
                # 파일명에서 인덱스명과 모델키 추출
                base_name = filename.replace('.faiss', '')
                # 마지막 '_'를 기준으로 분리
                parts = base_name.rsplit('_', 1)
                if len(parts) == 2:
                    index_name, model_key = parts
                    
                    # 메타데이터 정보 로드
                    metadata_path = os.path.join(self.save_directory, f"{base_name}_metadata.pkl")
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                        except:
                            pass
                    
                    if index_name not in saved_indexes:
                        saved_indexes[index_name] = {}
                    
                    saved_indexes[index_name][model_key] = {
                        'file_path': os.path.join(self.save_directory, filename),
                        'created_at': metadata.get('created_at', 'Unknown'),
                        'document_count': metadata.get('document_count', 'Unknown'),
                        'model_name': metadata.get('model_name', 'Unknown')
                    }
        
        return saved_indexes
    
    def delete_saved_index(self, index_name: str, model_key: str) -> str:
        """저장된 인덱스 삭제"""
        try:
            index_path = os.path.join(self.save_directory, f"{index_name}_{model_key}")
            metadata_path = os.path.join(self.save_directory, f"{index_name}_{model_key}_metadata.pkl")
            
            # FAISS 관련 파일들 삭제
            files_to_delete = [
                f"{index_path}.faiss",
                f"{index_path}.pkl",
                metadata_path
            ]
            
            deleted_files = []
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
            
            if deleted_files:
                return f"✅ 인덱스 삭제 완료: {', '.join(deleted_files)}"
            else:
                return f"❌ 삭제할 파일이 없습니다: {index_name}_{model_key}"
                
        except Exception as e:
            return f"❌ 인덱스 삭제 실패: {str(e)}"
    
    def auto_save_after_creation(self, documents: List[Document], index_name: str, model_key: str = None) -> FAISS:
        """문서로부터 벡터 스토어 생성 후 자동 저장"""
        vector_store = self.create_vector_store(documents, model_key)
        save_result = self.save_vector_store(index_name, model_key)
        print(save_result)
        return vector_store


if __name__ == "__main__":
    vector_store_manager = VectorStoreManager()