import React, { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";
import { poseSimilarity } from "posenet-similarity";
import { useCounter } from "./useCounter";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const posenetRef = useRef(false);
  const [distance, setDistance] = useState(0);
  const [maxDistance, setMaxDistance] = useState(distance);
  const { count: startCounter, start } = useCounter();
  const { count: breakCounter, start: startBreakCounter } = useCounter();

  /**
   * @param {posenet.PoseNet} net
   */
  const detect = useCallback(async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Make Detections
      const expectedPose = await net.estimateSinglePose(imgRef.current);
      const pose = await net.estimateSinglePose(video);
      const data = poseSimilarity(expectedPose, pose, {
        strategy: "cosineSimilarity",
      });
      setDistance(data);
      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
      window.requestAnimationFrame(() => detect(net));
    }
  }, []);
  //  Load posenet
  const runPosenet = useCallback(async () => {
    posenetRef.current = true;
    const net = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });

    //
    // setInterval(() => {
    window.requestAnimationFrame(() => detect(net));
    // }, 1000);
  }, [detect]);

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;
    drawKeypoints(pose["keypoints"], 0.6, ctx);
    drawSkeleton(pose["keypoints"], 0.7, ctx);
  };

  useEffect(() => {
    if (!posenetRef.current) {
      runPosenet();
    }
  }, [runPosenet]);

  useEffect(() => {
    if (distance > maxDistance) {
      setMaxDistance(distance);
    }
  }, [distance, maxDistance]);

  return (
    <div className="App">
      <div className="centered counter-container">
        {startCounter !== -1 ? (
          <div className="counter">{startCounter}</div>
        ) : (
          <button onClick={() => start(10)}>Start Posing</button>
        )}
      </div>
      <header
        className="App-header"
        style={{
          visibility:
            startCounter === 0 || breakCounter === 0 ? "visible" : "hidden",
        }}
      >
        <Webcam ref={webcamRef} className="centered" />
        <canvas ref={canvasRef} className="centered" />
        <div className="image-score">
          <img
            ref={imgRef}
            alt="pose"
            crossOrigin="anonymous"
            src="https://ucarecdn.com/fd629ccd-c734-4b1d-9659-4381458e0857/-/preview/-/quality/smart/-/format/auto/"
            id="pose-match"
            style={{ width: "100%" }}
          />
          <h1>{maxDistance.toFixed(2)}</h1>
        </div>
      </header>
    </div>
  );
}

export default App;
