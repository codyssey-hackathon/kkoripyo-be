# 꼬리표 (kkoripyo) 백엔드

SNS 공개정보와 관계망을 분석해 **제3자가 나에 대해 어디까지 추론할 수 있는지** 보여주는 서비스의 백엔드입니다.

- Python 3.12 · FastAPI · SQLAlchemy · SQLite
- CI: GitHub Actions (린트·테스트)
- CD: 서버에 설치된 self-hosted 러너가 `main` 푸시마다 Docker로 재배포

## 로컬 실행

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --reload
```

- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

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
  api/routes/          엔드포인트 (health, analyses)
tests/                 pytest
.github/workflows/     CI/CD
```

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
