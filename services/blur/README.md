# Text Blur Server

사진을 보내면 글자가 있는 부분만 블러 처리해서 돌려주는 API 서버입니다.
`Text-Blur-Florence2` 워크플로를 화면 없는(headless) ComfyUI에서 그대로 실행합니다.

```
클라이언트 ──이미지──▶ blur_server.py (:8190, 외부 공개)
                         │
                         ▼
                    ComfyUI (:8188, 127.0.0.1 전용, 화면 없이 사용)
                    Florence-2로 글자 위치 검출 → 블러 → 합성
```

- 요청이 끝나면 ComfyUI에 올라간 입력·출력 이미지와 작업 기록을 바로 지웁니다.
- ComfyUI는 서버 내부(127.0.0.1)에서만 열리고, 외부에는 블러 API만 노출됩니다.

## 파일

| 파일 | 내용 |
|---|---|
| `workflow_api.json` | Text-Blur-Florence2 워크플로 (API 형식) |
| `blur_server.py` | 블러 API 서버 |
| `setup_linux.sh` | ComfyUI, Florence-2 노드, 모델 설치 (한 번만 실행) |
| `run_linux.sh` | ComfyUI와 블러 API를 함께 실행 |

## 설치 (Linux + NVIDIA GPU)

필요한 것: Python 3.10 이상, git, curl, NVIDIA 드라이버

```bash
bash setup_linux.sh
```

- ComfyUI 0.41.2, Florence-2 노드, Florence-2-large 모델(약 1.5GB)을 이 폴더 안에 설치합니다.
- 버전은 원래 PC와 똑같이 고정되어 있습니다.
- 기본값은 CUDA 13.0용 PyTorch입니다. 서버 드라이버가 더 오래됐다면 바꿔서 실행하세요.
  ```bash
  TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_linux.sh
  ```

## 실행

```bash
bash run_linux.sh
```

API 키를 걸고 싶으면:
```bash
API_KEY=원하는키 bash run_linux.sh
```

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `8190` | 블러 API 포트 |
| `HOST` | `0.0.0.0` | 블러 API 주소 |
| `API_KEY` | 없음 | 설정하면 `X-API-Key` 헤더가 필요 |
| `TIMEOUT` | `300` | 이미지 한 장당 최대 대기 시간(초) |
| `COMFY_PORT` | `8188` | 내부 ComfyUI 포트 |

## 사용법

```bash
curl -X POST -F "image=@photo.jpg" http://서버주소:8190/blur -o blurred.png
```

- 마스크 여유를 바꾸려면 `?expand=20` (기본 12px)
- API 키를 설정했다면 `-H "X-API-Key: 원하는키"` 추가
- 상태 확인: `GET /health`

Python 예시:
```python
import requests
r = requests.post("http://서버주소:8190/blur", files={"image": open("photo.jpg", "rb")})
open("blurred.png", "wb").write(r.content)
```

## 이미지 링크로 여러 장 블러 (`POST /blur/urls`)

이미지 링크를 최대 4개 보내면, 이미지마다 블러된 사진을 JSON으로 돌려줍니다.
비포는 `image_url`, 애프터는 `blurred_image`를 그대로 `<img src>`에 쓰면 됩니다.

```bash
curl -X POST https://blur.purplecodelabs.com/blur/urls -H 'Content-Type: application/json' \
  -d '{"images":[{"image_id":"DXa1b2c3_0","image_url":"https://..."}]}'
```

```json
{"images": [
  {"image_id": "DXa1b2c3_0", "image_url": "https://...",
   "blurred_image": "data:image/jpeg;base64,/9j/...", "error": null}
]}
```

- 한 번에 최대 4장 (넘으면 400)
- 한 장이 실패해도 나머지는 처리하고, 실패한 장은 `blurred_image: null`, `error`에 사유
  (`fetch_failed`, `too_large`, `blur_failed`)
- 원본 이미지는 최대 10MB, 결과는 원본과 같은 해상도의 JPEG

## 도커로 실행 (GPU 서버)

```bash
docker compose up -d --build     # 처음 시작할 때 모델(약 1.5GB)을 볼륨에 받음
```

- ComfyUI는 컨테이너 안(127.0.0.1)에서만 열리고, 밖으로는 8190만 나갑니다.
- PyTorch CUDA 빌드는 기본 cu130 (드라이버 580 이상). 다르면 `TORCH_INDEX=.../cu128 docker compose build`
- 운영: GPU 서버(10.1.2.21) `~/kkoripyo-blur`, 외부 주소 `https://blur.purplecodelabs.com`

## 참고

- 첫 요청은 모델을 GPU에 올리느라 느립니다. 이후에는 모델이 메모리에 남아 있어서 빠릅니다.
- 요청이 동시에 들어오면 ComfyUI가 한 장씩 차례로 처리합니다.
- Florence-2는 글자 위치는 잘 찾지만, 아주 작거나 많이 기울어진 글자는 놓칠 수 있습니다.
