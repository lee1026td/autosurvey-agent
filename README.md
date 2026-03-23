# AutoSurvey Agent + RAG

로컬 LLM 기반 자동 리서치(AutoSurvey)와 RAG 질의응답을 하나의 CLI로 실행하는 프로젝트입니다.

- AutoSurvey: 웹 검색 -> 문서 수집 -> 문서/배치 요약 -> 최종 리포트 생성
- RAG: 생성된 마크다운 문서를 벡터 인덱싱 후 대화형 질의응답
- 통합 실행: AutoSurvey 완료 후 바로 RAG 챗으로 진입

현재 메인 엔트리포인트는 `main.py`입니다.

## 1) 사전 준비

### 1.1 Python 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 주요 모듈 설명:

- `openai`:
  - 로컬 `llama-server`의 OpenAI 호환 API(`/v1/chat/completions`, `/v1/embeddings`)를 호출하는 클라이언트로 사용
  - `llm_client.py`에서 채팅/임베딩 공통 인터페이스 제공
- `chromadb`:
  - 임베딩 벡터 저장/검색용 로컬 벡터DB
  - `vector_store.py`에서 PersistentClient 기반 저장소로 사용
- `langchain-community`, `duckduckgo-search`:
  - DuckDuckGo 검색 결과 수집에 사용
  - `autosurvey_tools.py`의 검색 단계에서 사용
- `requests`, `beautifulsoup4`:
  - 웹 페이지 다운로드 및 본문 추출/정제에 사용
  - 수집 문서를 텍스트/HTML로 저장하는 데 사용

### 1.2 llama-server 2개 실행 (권장)

이 프로젝트는 채팅 LLM 서버와 임베딩 서버를 분리해서 돌릴 수 있습니다.

예시:

- Chat LLM 서버: `127.0.0.1:8080`
- Embedding 서버: `127.0.0.1:8081`

```bash
# 터미널 1: Chat 모델 서버
llama-server -m ./models/chat-model.gguf -c 16384 -ngl -1 --port 8080

# 터미널 2: Embedding 모델 서버
llama-server -m ./models/embed-model.gguf -c 8192 -ngl -1 --port 8081
```

설명:

- Chat 서버는 계획/요약/최종 리포트/RAG 답변 생성에 사용
- Embedding 서버는 문서 및 질의 임베딩 생성에 사용
- 분리 실행 시 `main.py`에서 `--embed-host`, `--embed-port`를 함께 지정

참고:

- 임베딩 서버를 지정하지 않으면 chat 서버를 임베딩에도 같이 사용합니다.

## 2) RAG 시스템 개요

RAG 흐름은 아래와 같습니다.

1. 마크다운 문서(`final.md`, `summary/doc_*.md` 등)를 재귀적으로 탐색
2. 문서를 청크 단위로 분할(`rag_system.py`)
3. 임베딩 생성(`llm_client.py`)
4. ChromaDB에 저장(`vector_store.py`)
5. 사용자 질문을 임베딩 후 유사 문서 검색
6. 검색 결과를 컨텍스트로 LLM 답변 생성

핵심 포인트:

- 검색 결과 개수는 `--rag-results`로 조절
- 인덱스가 없으면 자동 인덱싱
- `--reindex`로 강제 재인덱싱 가능

## 3) 파일별 역할

- `main.py`
  - 통합 CLI 엔트리포인트
  - AutoSurvey 단계 실행 + RAG 모드 진입 제어
  - `--phase rag`, `--no-rag`, `--embed-host/--embed-port` 등 통합 옵션 제공
- `autosurvey_agent.py`
  - AutoSurvey 파이프라인 본체
  - Plan/Collect/Summarize/Final 단계 처리
  - 문서 중복 판정, 인덱스/요약/최종 리포트 파일 생성
- `autosurvey_tools.py`
  - 웹 검색, 페이지 fetch, 본문 추출, 중복 유사도 계산 등 유틸
- `llm_client.py`
  - llama-server OpenAI 호환 클라이언트 래퍼
  - `ask`, `ask_json`, `embed`, `embed_batch` 제공
- `rag_system.py`
  - 문서 청킹/인덱싱/검색/답변 생성 및 대화 루프 처리
- `vector_store.py`
  - ChromaDB 저장/조회/삭제 인터페이스
- `requirements.txt`
  - 프로젝트 실행에 필요한 Python 의존성 목록
- `deprecated/`
  - 이전 버전 코드 보관 디렉터리 (현재 기본 진입 경로 아님)

## 4) main.py 실행 인자

```text
positional:
  instruction                 자연어 리서치 요청 (옵션, 모드에 따라 필수)

required:
  --output-dir OUTPUT_DIR     결과 저장 루트 디렉터리

LLM:
  --host HOST                 chat llama-server host (default: 127.0.0.1)
  --port PORT                 chat llama-server port (default: 8080)
  --embed-host EMBED_HOST     임베딩 서버 host (분리 시 지정)
  --embed-port EMBED_PORT     임베딩 서버 port (분리 시 지정)

AutoSurvey:
  --batch-size BATCH_SIZE     배치 요약 크기 (default: 5)
  --max-docs MAX_DOCS         최대 수집 문서 수 (default: 15)
  --max-context MAX_CONTEXT   LLM 입력 최대 컨텍스트 (default: 16384)
  --force-plan                plan.json 강제 재생성
  --overwrite-summaries       기존 요약 강제 재생성

Phase:
  --phase {all,plan,collect,summarize,final,rag}
                              실행 단계 선택 (default: all)

Reasoning/Streaming:
  --plan-reasoning / --no-plan-reasoning
  --summary-reasoning / --no-summary-reasoning
  --final-reasoning / --no-final-reasoning
  --stream-summary
  --stream-reasoning
  --no-trace-latency

RAG:
  --no-rag                    all 완료 후 RAG 자동 진입 비활성화
  --rag-results RAG_RESULTS   RAG 검색 문서 수 (default: 5)
  --reindex                   벡터 인덱스 강제 재생성
```

## 5) 실행 시나리오별 예시

### 5.1 RAG만 사용

조건:

- `--output-dir` 아래에 `.md` 문서가 이미 존재

방법 A: 명시적으로 RAG 모드

```bash
python main.py \
  --output-dir ./runs/research_001 \
  --phase rag \
  --host 127.0.0.1 --port 8080 \
  --embed-host 127.0.0.1 --embed-port 8081
```

방법 B: 자동 진입 (instruction 없이 `--phase all` 기본값)

```bash
python main.py \
  --output-dir ./runs/research_001 \
  --host 127.0.0.1 --port 8080 \
  --embed-host 127.0.0.1 --embed-port 8081
```

- 이 경우 `output-dir` 내 markdown이 있으면 자동으로 RAG 모드로 전환됩니다.

### 5.2 AutoSurvey -> RAG 통합 사용

한 번의 실행으로 리서치를 끝내고 RAG 챗으로 바로 들어갑니다.

```bash
python main.py "최근 AI 에이전트 평가 방법론 동향" \
  --output-dir ./runs/research_agents \
  --host 127.0.0.1 --port 8080 \
  --embed-host 127.0.0.1 --embed-port 8081 \
  --max-docs 15 \
  --batch-size 5
```

동작:

1. plan -> collect -> summarize -> final 수행
2. 완료 후 자동으로 RAG chat_loop 진입 (`--no-rag` 미지정 시)

### 5.3 AutoSurvey만 사용

리서치 산출물만 만들고 종료하려면 `--no-rag`를 사용합니다.

```bash
python main.py "온디바이스 LLM 추론 최적화 전략" \
  --output-dir ./runs/research_onnx \
  --host 127.0.0.1 --port 8080 \
  --embed-host 127.0.0.1 --embed-port 8081 \
  --phase all \
  --no-rag
```

또는 단계별 실행도 가능합니다.

```bash
python main.py "제로트러스트 보안 모델" --output-dir ./runs/research_zero --phase plan
python main.py --output-dir ./runs/research_zero --phase collect
python main.py --output-dir ./runs/research_zero --phase summarize
python main.py --output-dir ./runs/research_zero --phase final
```

## 6) 출력 디렉터리 구조

```text
runs/
└── research_001/
    ├── final.md
    ├── chromadb/
    │   └── ...
    ├── corpus/
    │   ├── raw_html/
    │   │   ├── 000.html
    │   │   └── ...
    │   └── raw_text/
    │       ├── 000.txt
    │       └── ...
    └── summary/
        ├── request.txt
        ├── plan.json
        ├── index.json
        ├── doc_000.md
        ├── batch_001.md
        └── ...
```

## 7) 자주 발생하는 이슈

### 모델 연결 실패

오류 예:

```text
RuntimeError: No models available from llama-server.
```

점검:

- chat 서버 주소(`--host`, `--port`) 확인
- embedding 분리 시 `--embed-host`, `--embed-port` 확인
- 각 서버에서 모델이 정상 로드되었는지 확인

### 인덱싱이 비어 있음

- `--output-dir` 아래 markdown 파일 존재 여부 확인
- 강제로 다시 만들려면 `--phase rag --reindex`

### 수집/요약 품질 조절

- 더 빠르게: `--no-plan-reasoning --no-summary-reasoning --no-final-reasoning`
- 더 자세히: `--summary-reasoning --stream-summary --stream-reasoning`

## License

MIT License
