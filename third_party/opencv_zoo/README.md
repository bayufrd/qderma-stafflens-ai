# OpenCV Zoo: YuNet Face Detection & SFace Face Recognition (Apache-2.0)

This directory vendors the commercial-free **YuNet** face detector and **SFace** face recognizer pipeline from [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo).

## License
- **Code & Pretrained ONNX Models:** **Apache-2.0** (100% free for commercial, SaaS, and proprietary use).
  - `face_detection_yunet_2023mar.onnx` (Apache-2.0) — [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  - `face_recognition_sface_2021dec.onnx` (Apache-2.0) — [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

## Features
- Zero non-commercial restrictions (replaces InsightFace `buffalo_l` / `buffalo_s` research-only models).
- Built-in native OpenCV DNN acceleration (`cv2.FaceDetectorYN` + `cv2.FaceRecognizerSF`) producing L2-normalized 128-D embeddings.
