# VLM Study - 

## FastVLM 논문 정리 

### 목차
1. [용어 요약](#용어-요약)
2. [논문 내용 요약 (FastVLM)](#논문-내용-요약-fastvlm)

---

### 용어 요약

#### 1. ViT (Vision Transformer)
- **정의**: 이미지를 패치 단위로 분할하여 Transformer로 처리하는 모델
- **특징**: CNN 없이 pure attention 메커니즘 사용
- **문제점**: 고해상도에서 토큰 수 폭증 (O(n²) 복잡도)

#### 2. CLIP (Contrastive Language-Image Pre-training)
- **정의**: 이미지-텍스트 쌍을 contrastive learning으로 학습하는 멀티모달 모델
- **핵심**: 두 모달리티를 동일한 임베딩 공간에 매핑
- **활용**: VLM의 vision encoder로 널리 사용

#### 3. VLM (Vision Language Model)
- **구조**: Vision Encoder + Projection Layer + Large Language Model
- **예시**: LLaVA, GPT-4V, Flamingo 등
- **과제**: 고해상도 이미지 처리 시 효율성 문제

#### 4. TTFT (Time-To-First-Token)
- **정의**: Vision encoder 지연시간 + LLM prefilling 시간
- **중요성**: VLM의 실시간 성능을 결정하는 핵심 지표

#### 5. Hybrid Architecture
- **개념**: CNN과 Transformer를 결합한 구조
- **장점**: CNN의 local feature 추출 + Transformer의 global modeling

---

### 논문 내용 요약 (FastVLM)

#### 1. 연구 배경 및 동기

##### 문제 정의
- **해상도 확장의 필요성**: 텍스트가 많은 이미지, 차트 등의 이해를 위해 고해상도 처리 필수
- **기존 방법의 한계**: ViT 기반 vision encoder는 고해상도에서 토큰 수가 제곱에 비례해 증가
- **효율성 문제**: Vision encoding 지연 + LLM prefilling 지연으로 TTFT 급증

##### 연구 목표
고해상도 이미지 처리에서 **정확도는 유지하면서 효율성을 대폭 개선**하는 VLM 개발

#### 2. 제안 방법: FastVLM

##### 2.1 FastViTHD 아키텍처 
**5단계 하이브리드 설계:**
- Stage 1-3: RepMixer 블록 (효율적 convolution)
- Stage 4-5: Multi-head self-attention
- **핵심 혁신**: 64배 다운샘플링 (기존 16배 → 4배 적은 토큰 생성)

**아키텍처 세부사항:**
- 모델 깊이: [2, 12, 24, 4, 2]
- 임베딩 차원: [96, 192, 384, 768, 1536]
- 총 파라미터: 125.1M (ViT-L/14보다 2.4배 작음)

##### 2.2 Multi-Scale Feature 활용
- 서로 다른 스테이지의 특징을 집계
- Depthwise convolution을 통한 특징 결합
- 고수준 특징 보완으로 성능 향상

##### 2.3 해상도 스케일링 전략
**Static vs Dynamic Resolution:**
- **Static**: 모델 입력 해상도 직접 조정 (선호)
- **Dynamic**: 이미지 타일링 후 처리 (극한 해상도에서만 유리)

#### 3. 핵심 실험 결과

##### 3.1 효율성 대폭 개선
**LLaVA-OneVision 대비 (같은 0.5B LLM 사용):**
- **TTFT**: 85배 빠름
- **Vision encoder 크기**: 3.4배 작음
- **성능**: SeedBench, MMMU, DocVQA에서 우수

##### 3.2 정확도-지연시간 트레이드오프 최적화
**Pareto Optimal Curve 분석:**
- 다양한 (해상도, LLM 크기) 조합 실험
- FastViTHD가 FastViT보다 모든 구간에서 우수
- 동일 성능 달성 시 최대 3배 빠른 추론

##### 3.3 기존 방법들과의 비교
**Token Pruning 방법들과 비교:**
- 계층적 백본이 token pruning보다 우수한 트레이드오프
- FastViTHD 256×256이 기존 pruning 방법들보다 성능 우수

#### 4. 기술적 혁신점

##### 4.1 Vision-Language 상호작용 최적화
**체계적 분석:**
- 이미지 해상도, 비전 토큰 수, LLM 크기의 상호작용
- 실제 하드웨어(M1 MacBook Pro)에서 성능 측정
- 이론적 추정이 아닌 실측 기반 최적화

##### 4.2 확장성 입증
**데이터셋 스케일링:**
- Stage 1.5: 15M 샘플로 해상도 적응
- Stage 2: 1.1M~12.5M instruction tuning
- 데이터 증가에 따른 일관된 성능 향상

#### 5. 실용적 기여

##### 5.1 모바일/엣지 환경 적합성
- 실제 디바이스에서의 벤치마크 제공
- 자원 제약 환경에서의 VLM 배포 가능성 제시

##### 5.2 오픈소스 기여
- 모델 체크포인트 공개
- 재현 가능한 실험 설정 제공

#### 6. 한계점 및 향후 연구

##### 6.1 현재 한계
- 단일 vision encoder 의존 (앙상블 대비 한계 존재)
- 일부 fine-grained 시각적 추론에서 개선 여지

##### 6.2 미래 방향
- 더 효율적인 attention 메커니즘
- 동적 해상도 최적화 개선
- 다양한 모달리티 확장

#### 7. 결론 및 의의

##### 주요 기여
1. **실용적 효율성**: 실제 하드웨어에서 검증된 대폭적인 성능 개선
2. **설계 철학**: 복잡한 후처리 대신 아키텍처 최적화로 문제 해결
3. **확장성**: 다양한 해상도와 모델 크기에서 일관된 개선

##### 학술적 의의
- VLM 효율성 연구의 새로운 방향 제시
- 하이브리드 아키텍처의 우수성 입증
- 실용적 멀티모달 AI 시스템 구축에 기여

---

### 참고 자료

- **원본 논문**: [FastVLM: Efficient Vision Encoding for Vision Language Models](https://arxiv.org/html/2412.13303v2)
- **코드 및 모델**: [https://github.com/apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)

---

