import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as faceapi from "@vladmandic/face-api";
import { drawMesh } from "./drawMesh";

export const runDetector = async (
  video,
  canvas,
  setEmotions,
  setBlinkCount
) => {
  const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  const detectorConfig = {
    runtime: "tfjs",
  };
  const detector = await faceLandmarksDetection.createDetector(
    model,
    detectorConfig
  );

  let irisC = [];
  let nowBlinking = false;
  let blinkCount = 0;

  const detect = async (net) => {
    const estimationConfig = { flipHorizontal: false };
    const faces = await net.estimateFaces(video, estimationConfig);
    const ctx = canvas.getContext("2d");

    if (faces.length > 0) {
      const face = faces[0];
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions();
      if (detections.length > 0) {
        const expressions = detections[0].expressions;
        const emotionList = Object.entries(expressions).map(
          (emotion) => `${emotion[0]}: ${emotion[1].toFixed(2)}`
        );
        setEmotions(emotionList);

        const landmarks = detections[0].landmarks.positions;
        const x_ = landmarks[37].x;
        const y_ = landmarks[37].y;
        const w_ = landmarks[38].x - landmarks[37].x;
        const h_ = landmarks[41].y - landmarks[37].y;

        const frame = ctx.getImageData(0, 0, video.width, video.height);
        const p_ =
          Math.floor(x_ + w_ / 2) + Math.floor(y_ + h_ / 2) * video.width;
        const v_ = Math.floor(
          (frame.data[p_ * 4 + 0] +
            frame.data[p_ * 4 + 1] +
            frame.data[p_ * 4 + 2]) /
            3
        );

        irisC.push(v_);
        if (irisC.length > 100) {
          irisC.shift();
        }

        let meanIrisC =
          irisC.reduce((sum, element) => sum + element, 0) / irisC.length;
        let vThreshold = 1.5;
        let currentIrisC = irisC[irisC.length - 1];

        if (irisC.length === 100) {
          if (!nowBlinking && currentIrisC >= meanIrisC * vThreshold) {
            nowBlinking = true;
          } else if (nowBlinking && currentIrisC < meanIrisC * vThreshold) {
            nowBlinking = false;
            blinkCount += 1;
            setBlinkCount(blinkCount);
          }
        }
      }
      requestAnimationFrame(() => drawMesh(face, ctx));
    }

    detect(detector);
  };
  detect(detector);
};
