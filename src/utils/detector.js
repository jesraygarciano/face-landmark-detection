import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as faceapi from "@vladmandic/face-api";
import { drawMesh } from "./drawMesh";

export const runDetector = async (video, canvas, setEmotions) => {
  const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  const detectorConfig = {
    runtime: "tfjs",
  };
  const detector = await faceLandmarksDetection.createDetector(
    model,
    detectorConfig
  );

  const detect = async (net) => {
    const estimationConfig = { flipHorizontal: false };
    const faces = await net.estimateFaces(video, estimationConfig);
    const ctx = canvas.getContext("2d");

    if (faces.length > 0) {
      const face = faces[0];
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceExpressions();
      if (detections.length > 0) {
        const expressions = detections[0].expressions;
        const sortedEmotions = Object.entries(expressions).sort(
          (a, b) => b[1] - a[1]
        );
        setEmotions(
          sortedEmotions.map(
            (emotion) => `${emotion[0]}: ${emotion[1].toFixed(2)}`
          )
        );
      }
      requestAnimationFrame(() => drawMesh(face, ctx));
    }

    detect(detector);
  };
  detect(detector);
};
