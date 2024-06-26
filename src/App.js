import React, { useRef, useState, useEffect } from "react";
import "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@mediapipe/face_mesh";
import Webcam from "react-webcam";
import * as faceapi from "@vladmandic/face-api";
import { runDetector } from "./utils/detector";

const inputResolution = {
  width: 1080,
  height: 900,
};
const videoConstraints = {
  width: inputResolution.width,
  height: inputResolution.height,
  facingMode: "user",
};

function App() {
  const canvasRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [emotions, setEmotions] = useState([]);
  const [blinkCount, setBlinkCount] = useState(0);

  useEffect(() => {
    const loadModels = async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
      await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
      await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
      await faceapi.nets.faceExpressionNet.loadFromUri("/models");
    };
    loadModels();
  }, []);

  const handleVideoLoad = (videoNode) => {
    const video = videoNode.target;
    if (video.readyState !== 4) return;
    if (loaded) return;
    runDetector(video, canvasRef.current, setEmotions, setBlinkCount);
    setLoaded(true);
  };

  return (
    <div>
      <Webcam
        width={inputResolution.width}
        height={inputResolution.height}
        style={{ visibility: "hidden", position: "absolute" }}
        videoConstraints={videoConstraints}
        onLoadedData={handleVideoLoad}
      />
      <canvas
        ref={canvasRef}
        width={inputResolution.width}
        height={inputResolution.height}
        style={{ position: "absolute" }}
      />
      {loaded ? (
        <div>
          {emotions.map((emotion, index) => (
            <div key={index}>{emotion}</div>
          ))}
          <div>Blinks: {blinkCount}</div>
        </div>
      ) : (
        <header>Loading...</header>
      )}
    </div>
  );
}

export default App;
