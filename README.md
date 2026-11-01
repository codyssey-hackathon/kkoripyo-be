# 꼬리표 (kkoripyo) 백엔드

SNS 공개정보와 관계망을 분석해 **제3자가 나에 대해 어디까지 추론할 수 있는지** 보여주는 서비스의 백엔드입니다.

- Python 3.12 · FastAPI · SQLAlchemy · SQLite
- CI: GitHub Actions (린트·테스트)
- CD: 서버에 설치된 self-hosted 러너가 `main` 푸시마다 Docker로 재배포

## 함께 쓰는 저장소

| 저장소 | 역할 |
|---|---|
| [kkoripyo-be](https://github.com/codyssey-hackathon/kkoripyo-be) (이 저장소) | 수집·분석 에이전트·SSE API |
| [kkoripyo-fe](https://github.com/codyssey-hackathon/kkoripyo-fe) | 진단 화면 (React) |
| [kkoripyo-blur](https://github.com/codyssey-hackathon/kkoripyo-blur) | 사진 속 글자 블러 서버 (Florence-2, GPU) |
| [kkoripyo-OCR](https://github.com/codyssey-hackathon/kkoripyo-OCR) | 사진 속 글자 인식 서버 (PaddleOCR, GPU) |

OCR·블러 서버 주소가 비어 있거나 꺼져 있어도 분석은 계속됩니다(해당 기능만 빠짐).

## 실행 방법

### 1. 설치

필요: Python 3.12

```bash
git clone https://github.com/codyssey-hackathon/kkoripyo-be.git
cd kkoripyo-be
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

### 2. 환경설정

```bash
cp .env.example .env
```

`.env`에 아래 값을 채웁니다. API 키는 저장소에 올리지 않습니다.

| 변수 | 필수 | 설명 |
|---|---|---|
| `EVERYINFRA_API_KEY` | O | 인스타그램 공개 데이터 API 키 ([everyinfra.com](https://everyinfra.com)) |
| `OPENAI_API_KEY` | O | 분석 에이전트용 OpenAI API 키 |
| `OPENAI_MODEL` | | 기본 `gpt-6.1-sol` |
| `OPENAI_REASONING_EFFORT` | | 추론 강도 `low`·`medium`·`high`·`xhigh`, 기본 `high` (빈 값이면 넘기지 않음) |
| `OCR_SERVER_URL` | | 사진 글자 인식 서버 주소 (비우면 OCR 생략) |
| `BLUR_SERVER_URL` | | 사진 글자 블러 서버 주소 (비우면 블러 생략) |
| `CORS_ORIGINS` | | 프론트 주소 목록, 기본 `["http://localhost:3000"]` |
| `DEMO_ENABLED` | | 운영에서도 진단 데모 화면 열기 (로컬은 항상 열림) |
| `DATABASE_URL` | | 기본 SQLite `./data/kkoripyo.db` |

### 3. 실행

```bash
.venv/bin/uvicorn app.main:app --reload
```

- 진단 데모 화면: http://localhost:8000/api/v1/agent/demo (아이디를 넣으면 실시간 진단)
- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health
- 진단 API: `POST /api/v1/agent/stream` `{"handle": "인스타그램 아이디"}` → SSE

Docker로 실행할 때:

```bash
cp .env.example .env   # 값 채우기
docker compose -p kkoripyo up -d --build
```

## 테스트·린트

```bash
.venv/bin/pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## 폴더 구조

```
app/
  main.py              FastAPI 앱, 라우터 등록, 시작 시 테이블 생성
  core/config.py       환경변수 설정 (.env)
  core/db.py           DB 엔진·세션
  models/entities.py   분석·계정·게시물·상호작용·단서·가설·근거·에이전트 로그 테이블
  api/router.py        /api/v1 라우터 묶음
  api/routes/          엔드포인트 (health, analyses, agent)
  integrations/everyinfra.py  EveryInfra API 호출 (비동기 작업 대기·재시도 포함)
  collect/instagram.py        인스타 데이터 수집 + 필요한 필드만 남기는 정제
  agent/tools.py              에이전트 도구 (설명·입력 스키마 = 도구 사용법)
  agent/graph.py              LangGraph 에이전트 기본 틀
  agent/prompts.py            에이전트 시스템 프롬프트
tests/                 pytest
.github/workflows/     CI/CD
```

## 분석 에이전트 (LangGraph)

`.env`에 `EVERYINFRA_API_KEY`, `OPENAI_API_KEY`, `OPENAI_MODEL`을 넣어야 합니다.

- `POST /api/v1/agent/run` `{"handle": "아이디"}`: 에이전트가 도구로 데이터를 모으고 리포트를 씁니다.
- `GET /api/v1/instagram/{handle}`: LLM 없이 `collect_account` 결과만 확인합니다 (EveryInfra 요금 발생).

그래프는 가장 기본적인 "도구 쓰는 에이전트" 모양입니다.

```
START → agent ──(도구 호출 있음)──→ tools ─┐
          ↑                                │
          └────────────────────────────────┘
          └──(도구 호출 없음)──→ END (리포트)
```

- `agent` 노드: LLM이 답하거나 도구 호출을 요청합니다.
- `tools` 노드: 요청된 도구를 실행합니다. 한 번에 여러 개를 요청하면 동시에 실행합니다.
- 새 단계(예: 사진 블러)는 `graph.py`에서 `add_node` / `add_edge`로 이어 붙입니다.

| 도구 | 용도 |
|---|---|
| `collect_account` | 대상 계정의 프로필·팔로워·팔로잉·게시물·댓글을 한 번에 (내부에서 동시 요청) |
| `get_profile` | 다른 계정 프로필만 가볍게 |
| `get_followers` / `get_following` | 다른 계정의 팔로워/팔로잉 목록 |
| `get_user_posts` | 다른 계정의 최근 게시물 |
| `get_post_comments` | 게시물 하나의 댓글 |

EveryInfra 한계: 좋아요 누른 사람 목록 없음, 여러 장 게시물은 첫 장 사진만, 사진 주소는 약 5일 뒤 만료.

## CI/CD

`.github/workflows/ci-cd.yml`

| 이벤트 | 동작 |
|---|---|
| PR | 린트·테스트 |
| `main` 푸시 | 린트·테스트 → 통과하면 서버에서 `docker compose up -d --build` → 헬스체크 |
| 수동 실행 (Actions 탭) | 위와 동일 |

배포 서버는 내부망(10.1.1.21)이라 GitHub이 직접 접속할 수 없습니다. 그래서 서버에 GitHub Actions **self-hosted 러너**(라벨 `kkoripyo`)를 서비스로 설치해, 서버가 GitHub에서 배포 작업을 받아와 실행합니다.

### 서버 정보

- 외부 주소: **https://kkoripyo.purplecodelabs.com** (API 문서: `/docs`)
- 내부 주소: `http://10.1.1.21:8000` (컨테이너 `kkoripyo-api`, 데이터는 Docker 볼륨 `kkoripyo_kkoripyo-data`)
- 러너: `~/actions-runner` (systemd 서비스로 자동 시작)
- 외부 접속: Cloudflare Tunnel `kkoripyo` (서비스 `cloudflared`, 설정 `/etc/cloudflared/config.yml`)
  - 서버 네트워크가 UDP를 막아 `protocol: http2`로 고정
  - DNS는 `kkoripyo.purplecodelabs.com` CNAME 하나만 사용 (루트 도메인과 무관)
- 서버 전용 환경변수: `~/kkoripyo/.env`가 있으면 배포 때 자동 적용 (API 키 등은 여기에, 저장소에는 올리지 않음)

```bash
# 서버에서 자주 쓰는 명령
docker compose -p kkoripyo ps
docker logs -f kkoripyo-api
sudo systemctl status 'actions.runner.*'
sudo systemctl status cloudflared
```
