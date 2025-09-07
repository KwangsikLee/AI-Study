#!/usr/bin/env python3
"""
고등학생 학과 선택 도우미 - 설정 및 테스트 스크립트
환경 설정부터 RAG 시스템 테스트까지 전체 과정을 실행

Author: kwangsiklee
Version: 0.1.0
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")
    
    # 1. Python 버전 확인
    python_version = sys.version_info
    print(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    # 2. .env 파일 확인
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env 파일이 없습니다. .env.example을 참고하여 생성하세요.")
        return False
    
    # 3. 환경 변수 로드
    load_dotenv()
    
    # 4. API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return False
    
    print("✅ 환경 설정 확인 완료")
    return True

def install_dependencies():
    """의존성 패키지 설치"""
    print("📦 의존성 패키지 설치 중...")
    
    try:
        # requirements.txt 설치
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ 의존성 패키지 설치 완료")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def check_directories():
    """필요한 디렉토리 확인 및 생성"""
    print("📁 디렉토리 구조 확인 중...")
    
    directories = [
        "korea_univ_guides",
        "temp_images", 
        "vector_db"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"  📂 생성: {dir_name}/")
        else:
            print(f"  ✅ 존재: {dir_name}/")
    
    # PDF 파일 확인
    pdf_dir = Path("korea_univ_guides")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"  📄 PDF 파일: {len(pdf_files)}개")
    
    if len(pdf_files) == 0:
        print("  ⚠️ korea_univ_guides 폴더에 PDF 파일이 없습니다.")
        print("     대학교 안내 PDF 파일을 추가하세요.")
        return False
    
    print("✅ 디렉토리 구조 확인 완료")
    return True

def test_modules():
    """모듈 import 테스트"""
    print("🧪 모듈 import 테스트 중...")
    
    try:
        # 필수 모듈들 import 테스트
        import gradio
        print(f"  ✅ Gradio {gradio.__version__}")
        
        import openai
        print(f"  ✅ OpenAI")
        
        import langchain
        print(f"  ✅ LangChain")
        
        # a_my_rag_module 모듈 테스트
        sys.path.append(str(Path(__file__).parent.parent))
        from a_my_rag_module import PDFImageExtractor, KoreanOCR
        print(f"  ✅ a_my_rag_module")
        
        print("✅ 모듈 import 테스트 완료")
        return True
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False

def test_rag_system():
    """RAG 시스템 기본 테스트"""
    print("🤖 RAG 시스템 테스트 중...")
    
    try:
        from college_rag_system import CollegeRAGSystem
        
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
        
        # 벡터 스토어 상태 확인
        info = rag_system.get_vector_store_info()
        print(f"  📊 벡터 스토어 존재: {info['exists']}")
        
        print("✅ RAG 시스템 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ RAG 시스템 테스트 실패: {e}")
        return False

def main():
    """메인 설정 함수"""
    print("🎓 고등학생 학과 선택 도우미 - 환경 설정")
    print("=" * 60)
    
    # 단계별 확인
    steps = [
        ("환경 설정 확인", check_environment),
        ("디렉토리 구조 확인", check_directories),
        ("의존성 설치", install_dependencies),
        ("모듈 테스트", test_modules),
        ("RAG 시스템 테스트", test_rag_system)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 단계: {step_name}")
        print("-" * 40)
        
        if not step_func():
            failed_steps.append(step_name)
            print(f"❌ {step_name} 실패")
        else:
            print(f"✅ {step_name} 성공")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📋 설정 결과 요약")
    print(f"{'='*60}")
    
    if not failed_steps:
        print("🎉 모든 설정이 완료되었습니다!")
        print("\n다음 단계:")
        print("1. python main.py - Gradio UI 실행")
        print("2. python college_rag_system.py - RAG 시스템 직접 테스트")
        return True
    else:
        print(f"❌ 실패한 단계: {', '.join(failed_steps)}")
        print("\n문제를 해결한 후 다시 실행하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)