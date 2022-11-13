import React, { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import JSConfetti from "js-confetti";
import { drawKeypoints, drawSkeleton } from "./utilities";
import { poseSimilarity } from "posenet-similarity";
import { useCounter } from "./useCounter";
import { images } from "./constants";

const jsConfetti = new JSConfetti();
let requestId;
function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const posenetRef = useRef(null);
  const [distance, setDistance] = useState(0);
  const [maxDistance, setMaxDistance] = useState(distance);
  const { count: startCounter, start } = useCounter();
  const { count: breakCounter, start: startBreakCounter } = useCounter();
  const [currentImage, setCurrentImage] = useState(0);
  const [success, setSuccess] = useState(new Array(images.length).fill(false));

  /**
   * @param {posenet.PoseNet} net
   */
  const detect = useCallback(async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4 &&
      posenetRef.current
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      const net = posenetRef.current;

      // Make Detections
      const pose = await net.estimateSinglePose(video);
      const expectedPose = await net.estimateSinglePose(imgRef.current);
      const data = poseSimilarity(expectedPose, pose, {
        strategy: "cosineSimilarity",
      });
      setDistance(data);
      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
      requestId = window.requestAnimationFrame(() => detect());
    }
  }, []);
  //  Load posenet
  const runPosenet = useCallback(async () => {
    posenetRef.current = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    window.requestAnimationFrame(detect);
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

  // once the start counter ends or scores exceeds reset and start break of 5secs
  useEffect(() => {
    if (startCounter === 0 || maxDistance > 0.99) {
      setSuccess((value) => {
        value[currentImage] = maxDistance > 0.99;
        return value;
      });
      setMaxDistance(0);
      cancelAnimationFrame(requestId);
      startBreakCounter(5);
    }
  }, [maxDistance, startCounter, currentImage]); //eslint-disable-line

  // when break counter resets, move to next image and start again
  useEffect(() => {
    if (breakCounter === 0 && currentImage < images.length - 1) {
      setCurrentImage(currentImage + 1);
      detect();
      start(10);
    }
  }, [detect, breakCounter, currentImage]); //eslint-disable-line

  // show the success confetti
  useEffect(() => {
    if (success.every((value) => value)) {
      jsConfetti.addConfetti({ emojis: ["✨"] });
      setSuccess((value) => value.map((val) => false));
    }
  }, [success]);

  return (
    <div className="container">
      <div className="centered">
        <Webcam ref={webcamRef} />
        <canvas className="canvas" ref={canvasRef} />
      </div>
      <button
        onClick={() => {
          start(10);
        }}
        disabled={startCounter > -1}
      >
        {startCounter > 0 ? startCounter : "Start"}
      </button>
      <div
        className="image-score"
        style={{
          visibility: startCounter > -1 ? "visible" : "hidden",
        }}
      >
        <img
          ref={imgRef}
          alt="pose"
          crossOrigin="anonymous"
          src={images[currentImage]}
          id="pose-match"
          style={{ width: "100%" }}
        />
        <h1>{maxDistance.toFixed(2)}</h1>
      </div>
    </div>
  );
}

export default App;
