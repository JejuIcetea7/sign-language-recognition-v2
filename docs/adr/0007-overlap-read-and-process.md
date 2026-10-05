# 0007 영상 읽기와 MediaPipe 처리를 겹쳐서 실행

## Status  
Proposed  


## Context  
- 특징 추출은 24프레임을 모두 읽은 뒤 MediaPipe로 처리해서, 걸리는 시간이 두 구간의 합이었다. (영상 읽기 208 ms + MediaPipe 266 ms, [실험 0006](../experiments/0006-latency-24f-sequential.md))
- 읽는 스레드와 처리하는 스레드를 나눠 겹치는 방식을 비교했다. ([실험 0008](../experiments/0008-overlap-read-and-process.md))

| | p50 | p95 |
|---|---|---|
| 순차(기존) | 496.5 ms | 550.9 ms |
| 겹치기 | 407.2 ms | 452.7 ms |

- 서버 엔드투엔드(p50 / p95)는 486.3 / 538.6 ms에서 418.0 / 456.3 ms로 줄었다. 동시요청 처리량은 약 2.1 req/s에서 약 2.2 req/s로 늘었다.


## Decision  
메인 파이프라인의 특징 추출에서 영상 읽기와 MediaPipe 처리를 겹쳐서 실행한다.  
읽는 스레드가 프레임을 큐(크기 4)에 넣고, 메인 스레드가 꺼내는 대로 처리한다.


## Consequences  
- 특징은 그대로이고 특징 추출 시간이 약 18% 줄어든다.
- 읽기와 MediaPipe가 겹치므로 `extract_video_features` 안에서는 두 구간 시간을 따로 잴 수 없다. 구간별 측정(`benchmark.py stages`)은 `read_frames`와 `extract_frame_features`를 따로 호출해서 잰다.
- 서버가 요청을 하나씩 처리하는 구조는 그대로라서, 동시 접속이 늘면 지연이 비례해 늘어난다.
