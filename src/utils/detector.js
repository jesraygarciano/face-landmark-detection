import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as faceapi from "@vladmandic/face-api";
import { drawMesh } from "./drawMesh";

// Function to calculate Euclidean distance
const euclideanDistance = (point1, point2) => {
  const dx = point1.x - point2.x;
  const dy = point1.y - point2.y;
  return Math.sqrt(dx * dx + dy * dy);
};

// Function to calculate Eye Aspect Ratio (EAR)
const calculateEAR = (eye) => {
  const A = euclideanDistance(eye[1], eye[5]);
  const B = euclideanDistance(eye[2], eye[4]);
  const C = euclideanDistance(eye[0], eye[3]);
  return (A + B) / (2.0 * C);
};

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

  let blinkCount = 0;
  let blinkThreshold = 0.25; // Threshold for EAR to detect blink
  let consecutiveFrames = 5; // Number of consecutive frames to confirm a blink
  let frameCounter = 0;

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
        const leftEye = [
          landmarks[36],
          landmarks[37],
          landmarks[38],
          landmarks[39],
          landmarks[40],
          landmarks[41],
        ];
        const rightEye = [
          landmarks[42],
          landmarks[43],
          landmarks[44],
          landmarks[45],
          landmarks[46],
          landmarks[47],
        ];

        const leftEAR = calculateEAR(leftEye);
        const rightEAR = calculateEAR(rightEye);
        const ear = (leftEAR + rightEAR) / 2.0;

        if (ear < blinkThreshold) {
          frameCounter += 1;
        } else {
          if (frameCounter >= consecutiveFrames) {
            blinkCount += 1;
            setBlinkCount(blinkCount);
          }
          frameCounter = 0;
        }
      }
      requestAnimationFrame(() => drawMesh(face, ctx));
    }

    detect(detector);
  };
  detect(detector);
};
