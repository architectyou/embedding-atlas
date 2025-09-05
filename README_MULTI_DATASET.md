# 🗺️ Multi-Dataset Embedding Atlas

기존 embedding-atlas를 확장하여 여러 parquet 파일을 하나의 서버에서 접근할 수 있는 다중 데이터셋 시각화 도구입니다.

## 🚀 사용법

### 서버 실행
```bash
# conda 환경 활성화
conda activate lm-analysis

# 다중 데이터셋 서버 실행
python multi_atlas_cli.py
```

### 접속
서버 실행 후 브라우저에서 접속:

- **메인 페이지**: `http://localhost:5055/` (또는 표시된 포트)
- **데이터셋 목록**: `http://localhost:5055/datasets`

### 개별 데이터셋 시각화
각 데이터셋은 별도 경로에서 접근:

- **BGE FOMC (2015-08)**: `/dataset/bge_fomc_0815`
- **BGE FOMC (비전처리)**: `/dataset/bge_fomc_non_preprocessed`  
- **야구 스카우팅 (BGE-M3)**: `/dataset/scouting_report_bgem3`
- **야구 스카우팅 (OpenAI)**: `/dataset/scouting_report_openai`
- **야구 스카우팅 (Qwen-8B)**: `/dataset/scouting_report_qwen8b`

## ✨ 특징

- **자동 UMAP 계산**: 벡터가 있지만 좌표가 없으면 자동으로 2D 투영 계산
- **독립적 임베딩 공간**: 각 데이터셋이 고유한 임베딩 공간 유지
- **기존 품질 보장**: 원본 embedding-atlas의 모든 기능과 성능 유지
- **쉬운 비교**: 브라우저 탭으로 데이터셋 간 즉시 비교

## 🎯 비교 분석 사용례

### 임베딩 모델 비교
같은 야구 스카우팅 데이터에 대해 서로 다른 임베딩 모델의 결과 비교:
- BGE-M3 (1024차원)
- OpenAI (1536차원)
- Qwen-8B (4096차원)

### 전처리 효과 비교
FOMC 문서에서 전처리 유무에 따른 임베딩 차이:
- 전처리된 데이터 vs 원본 데이터

### 도메인별 분석
- 금융 문서 (FOMC) vs 스포츠 텍스트 (야구 스카우팅)
- 각 도메인에서 임베딩이 형성하는 클러스터 패턴 분석

## 🔧 설정 옵션

```bash
python multi_atlas_cli.py [OPTIONS]

Options:
  --host TEXT       Host address (default: localhost)
  --port INTEGER    Port number (default: 5055)
  --auto-port       Automatically find available port (default: enabled)
  --duckdb TEXT     DuckDB connection mode (default: wasm)
  --static TEXT     Custom static files path
  --help           Show help message
```

## 📁 지원 파일 형식

- **Parquet 파일**: 메인 데이터 형식
- **벡터 컬럼**: list 또는 numpy array 형식의 임베딩
- **텍스트 컬럼**: hover 정보용 (자동 감지: text, content, description 등)
- **좌표 컬럼**: 기존 x, y 좌표 (있으면 재사용, 없으면 UMAP 계산)

## 🎨 시각화 기능

각 데이터셋 페이지에서 제공되는 기능:
- 인터랙티브 줌/팬
- 포인트 hover로 텍스트 내용 확인
- 실시간 검색 및 필터링  
- 자동 밀도 클러스터링
- 근접 이웃 탐색
- 데이터 선택 및 내보내기

---

**참고**: 단일 데이터셋만 보려면 기존 embedding-atlas CLI를 사용하세요:
```bash
embedding-atlas your-data.parquet
```