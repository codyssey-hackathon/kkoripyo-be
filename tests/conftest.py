import os

os.environ["DATABASE_URL"] = "sqlite:///./data/test.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def no_real_ocr(monkeypatch):
    """테스트에서는 실제 OCR 서버를 부르지 않는다 (OCR 끔). OCR 테스트는 직접 가짜를 끼운다."""
    from app.integrations import ocr

    monkeypatch.setattr(ocr, "make_ocr_client", lambda: ocr.OcrClient(base_url=""))


@pytest.fixture(autouse=True)
def no_real_blur(monkeypatch):
    """테스트에서는 실제 블러 서버를 부르지 않는다."""
    from app.integrations import blur

    monkeypatch.setattr(blur, "make_blur_client", lambda: blur.BlurClient(base_url=""))
