# 0002. MediaPipe 버전을 1.0.0으로 고정


## Status  
Proposed  


## Context  
MediaPipe 1.0.1에서 손 인식 모델(`HandLandmarker`)을 만드는 순간 프로그램이 강제 종료된다. (영상에서 특징을 뽑는 `extract_features.py`와 서빙의 `predict.py`가 모두 막힘)

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
  -[DrishtiMetalHelper initWithCalculatorContext:]
  mediapipe::api2::TensorsToDetectionsCalculator::Open()
종료 코드 134
```

- 환경: macOS (Apple M5), Python 3.12 / 3.13
- 원인: 1.0.1이 GPU 서비스가 없는 상태에서도 Metal(GPU) 도우미를 무조건 만든다. 1.0.0 → 1.0.1에서 생긴 회귀로, [google-ai-edge/mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)과 같은 증상이다. 이슈는 닫혔지만 PyPI의 최신 배포판은 1.0.1이라 수정판이 없다.
- 효과 없던 시도: `Delegate.CPU` 지정, `MEDIAPIPE_DISABLE_GPU=1`, Python 버전 변경, 샌드박스 해제
- 같은 환경에서 비교: 1.0.1만 크래시, 1.0.0 / 0.10.30 / 0.10.33 / 0.10.35는 정상
- 10월 2일에는 1.0.1로도 동작했다. 그때는 GPU(Metal) 서비스가 있었던 것으로 추정한다. (로그에 Metal GL 컨텍스트 생성이 남아 있음)


## Decision  
`mediapipe==1.0.0`으로 고정한다. 정상 동작하는 버전 중 1.0.1과 가장 가깝다.


## Consequences  
- 홈서버를 포함해 모든 환경에서 같은 버전으로 설치한다. (의존성 목록 파일에 명시)
- 수정된 새 버전이 나오면 고정을 풀지 재검토한다.
- 확인: 1.0.0 설치 후 `predict.py`가 실제 영상으로 동작했고, 영상 44개(단어마다 2개)를 모두 맞혔다. 이 영상들은 학습에 쓴 것이라 성능이 아니라 특징 추출이 학습 때와 일관된다는 확인이다.
