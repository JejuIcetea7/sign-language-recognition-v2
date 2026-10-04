"""
- 영상을 업로드로 받아 수어 단어를 예측하는 API 서버
- 실행: python -m uvicorn --app-dir src server:app --port 8000
- 모델은 서버가 켜질 때 한 번만 불러온다 (모델 파일을 바꾸려면 환경변수 MODEL_PATH 지정)
"""

import os
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from models import load_model
from predict import default_model_path, predict_video

state = {}  # 서버가 켜질 때 불러온 모델과 라벨 목록
lock = threading.Lock()  # ponytail: 손 인식기를 한 번에 한 요청만 쓰도록 직렬 처리, 동시 요청이 많아지면 인식기를 요청마다 따로 만들기


@asynccontextmanager
async def lifespan(app):
    state["model"], state["classes"] = load_model(os.environ.get("MODEL_PATH", default_model_path))
    yield


app = FastAPI(lifespan=lifespan)


# 서버가 살아 있는지 확인
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


# 영상을 받아서 예측한 단어를 반환
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # cv2.VideoCapture는 파일 경로가 필요해서 업로드된 영상을 임시 파일로 저장
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        with lock:
            label = predict_video(tmp_path, state["model"], state["classes"])
    except ValueError as e:  # 영상이 아니거나 프레임을 못 읽어서 특징 모양이 (6, 111)이 아닌 경우
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {"label": label}
