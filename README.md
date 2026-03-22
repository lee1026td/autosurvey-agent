# AutoSurvey Agent

자동 웹 리서치 및 문서 수집, 요약 도구입니다. 사용자의 자연언어 질의를 기반으로 웹에서 자료를 수집하고, LLM을 활용하여 문서를 요약 및 종합하여 최종 리서치 리포트를 생성합니다.

## 사전 요구사항

### llama-server 실행

**중요:** AutoSurvey Agent를 사용하기 전에 llama-server를 미리 실행해야 합니다.

다음은 Qwen 3.5 모델을 예시로 한 llama-server 실행 방법입니다:

```bash
# Qwen 3.5 모델 사용
llama-server -m qwen-3.5.gguf -c 25000 -ngl -1
```

**옵션 설명:**
- `-m`: 사용할 모델 파일 (예: `qwen-3.5.gguf`)
- `-c 25000`: 컨텍스트 윈도우 크기 (최대 25,000 토큰)
- `-ngl -1`: GPU 레이어 오프로드 (-1은 모든 레이어를 GPU에서 처리)

llama-server는 기본적으로 `http://127.0.0.1:8080`에서 실행되며, AutoSurvey Agent가 이 주소로 연결합니다.

다른 모델을 사용하려면 모델의 권장 설정에 맞게 `-c` 옵션값을 조정하세요.

## 설치

```bash
pip install -r requirements.txt
```

필수 패키지:
- `requests` - HTTP 요청용
- `beautifulsoup4` - HTML 파싱용
- `openai` - LLM 클라이언트
- `langchain-community` - DuckDuckGo 검색 도구

## 사용법

### 1. 전체 파이프라인 실행 (기본)

```bash
python autosurvey_agent.py "당신의 리서치 질문" --output-dir ./runs/research_001
```

예시:
```bash
python autosurvey_agent.py "최근 AI 보안 위협 동향" --output-dir ./runs/research_001
```

이 명령은 다음의 모든 단계를 자동으로 실행합니다:
1. **Plan**: 리서치 계획 생성
2. **Collect**: 웹 문서 수집
3. **Summarize**: 문서 요약 및 배치 요약 생성
4. **Final**: 최종 리포트 생성

### 2. 단계별 실행

리서치의 각 단계를 개별적으로 실행할 수 있습니다.

#### 2.1 계획 수립 (`--phase plan`)

```bash
python autosurvey_agent.py "당신의 리서치 질문" --output-dir ./runs/research_001 --phase plan
```

예시:
```bash
python autosurvey_agent.py "클라우드 보안 Best Practices" --output-dir ./runs/research_001 --phase plan
```

**출력:**
- `./runs/research_001/summary/plan.json` - 검색 쿼리 및 리서치 목표 포함

#### 2.2 문서 수집 (`--phase collect`)

```bash
python autosurvey_agent.py --output-dir ./runs/research_001 --phase collect
```

또는 새로운 요청으로 수집:
```bash
python autosurvey_agent.py "새로운 리서치 주제" --output-dir ./runs/research_001 --phase collect --force-plan
```

**출력:**
- `./runs/research_001/corpus/raw_html/` - 원본 HTML 파일들
- `./runs/research_001/corpus/raw_text/` - 추출된 텍스트 파일들
- `./runs/research_001/summary/index.json` - 문서 메타데이터

#### 2.3 문서 요약 (`--phase summarize`)

```bash
python autosurvey_agent.py --output-dir ./runs/research_001 --phase summarize
```

이미 존재하는 요약을 덮어쓰려면:
```bash
python autosurvey_agent.py --output-dir ./runs/research_001 --phase summarize --overwrite-summaries
```

**출력:**
- `./runs/research_001/summary/doc_000.md` ~ `doc_014.md` - 각 문서별 요약
- `./runs/research_001/summary/batch_001.md` ~ - 배치 요약 (기본 5개 문서 단위)

#### 2.4 최종 리포트 생성 (`--phase final`)

```bash
python autosurvey_agent.py --output-dir ./runs/research_001 --phase final
```

**출력:**
- `./runs/research_001/final.md` - 최종 종합 리포트

## 주요 옵션

```
위치 인자:
  instruction              자연언어 리서치 요청 (필수, --phase all 또는 plan 실행 시)

옵션:
  --output-dir DIR         출력 디렉토리 (필수)
  --host HOST              llama-server 호스트 (기본: 127.0.0.1)
  --port PORT              llama-server 포트 (기본: 8080)
  --batch-size N           배치 요약 문서 수 (기본: 5)
  --max-docs N             최대 수집 문서 수 (기본: 15)
  --max-context N          LLM 입력 최대 토큰 (기본: 16384)
  --phase {all,plan,collect,summarize,final}  실행할 단계 (기본: all)
  --force-plan             기존 plan.json 덮어쓰기
  --overwrite-summaries    기존 요약 문서들 덮어쓰기
  --plan-reasoning         계획 수립에 추론 사용 (기본: true)
  --summary-reasoning      문서 요약에 추론 사용 (기본: false)
  --final-reasoning        최종 리포트에 추론 사용 (기본: true)
  --stream-summary         요약 생성 중 토큰 출력
  --stream-reasoning       추론 토큰 출력
  --no-trace-latency       LLM 응답 시간 로그 비활성화
```

## 출력 구조

```
runs/
└── research_001/
    ├── final.md                          # 최종 리포트
    ├── corpus/
    │   ├── raw_html/
    │   │   ├── 000.html
    │   │   ├── 001.html
    │   │   └── ...
    │   └── raw_text/
    │       ├── 000.txt
    │       ├── 001.txt
    │       └── ...
    └── summary/
        ├── plan.json                     # 리서치 계획
        ├── request.txt                   # 원본 요청
        ├── index.json                    # 문서 메타데이터
        ├── doc_000.md                    # 개별 문서 요약
        ├── doc_001.md
        ├── ...
        ├── batch_001.md                  # 배치 요약
        ├── batch_002.md
        └── ...
```

## 예시 워크플로우

### 예시 1: 완전 자동화 리서치

```bash
# llama-server 실행 (백그라운드)
llama-server -m qwen-3.5.gguf -c 25000 -ngl -1 &

# 전체 리서치 파이프라인 실행
python autosurvey_agent.py "Python의 최근 성능 최적화 기법" \
  --output-dir ./runs/python_perf \
  --max-docs 20 \
  --batch-size 5
```

### 예시 2: 단계별 리서치 (중단점 있음)

```bash
# 1. 계획 수립
python autosurvey_agent.py "기후 변화의 경제적 영향" \
  --output-dir ./runs/climate_economics \
  --phase plan

# (계획 검토)

# 2. 문서 수집
python autosurvey_agent.py \
  --output-dir ./runs/climate_economics \
  --phase collect \
  --max-docs 15

# (수집된 문서 확인)

# 3. 요약 생성
python autosurvey_agent.py \
  --output-dir ./runs/climate_economics \
  --phase summarize

# 4. 최종 리포트
python autosurvey_agent.py \
  --output-dir ./runs/climate_economics \
  --phase final
```

### 예시 3: 커스텀 LLM 설정

다른 호스트에서 llama-server를 실행하는 경우:

```bash
# llama-server가 192.168.1.100:8080에서 실행 중
python autosurvey_agent.py "당신의 질문" \
  --output-dir ./runs/research_001 \
  --host 192.168.1.100 \
  --port 8080
```

## 모드 및 옵션 조합

### 고속 모드 (추론 비활성화)

```bash
python autosurvey_agent.py "빠른 리서치" \
  --output-dir ./runs/fast \
  --plan-reasoning false \
  --summary-reasoning false \
  --final-reasoning false \
  --max-docs 10
```

### 상세 모드 (모든 추론 활성화)

```bash
python autosurvey_agent.py "상세 리서치" \
  --output-dir ./runs/detailed \
  --plan-reasoning true \
  --summary-reasoning true \
  --final-reasoning true \
  --stream-summary \
  --stream-reasoning \
  --max-docs 20
```

## 트러블슈팅

### llama-server 연결 실패
```
RuntimeError: No models available from llama-server.
```
- llama-server가 실행 중인지 확인하세요
- `--host`와 `--port` 설정을 확인하세요
- 기본 주소: `http://127.0.0.1:8080`

### 메모리 부족
```bash
# max-context 값을 줄이세요
python autosurvey_agent.py ... --max-context 8192
```

### 문서 수집 실패
- 네트워크 연결 확인
- 대상 웹사이트의 접근 제한 확인
- `--max-docs` 값을 줄여 실행

### 요약 품질 개선
```bash
# 추론 활성화 (더 느리지만 더 나은 품질)
python autosurvey_agent.py ... --summary-reasoning true

# 컨텍스트 크기 증가
python autosurvey_agent.py ... --max-context 24000
```

## 라이선스

MIT License
