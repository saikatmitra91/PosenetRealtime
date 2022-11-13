import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";
import { poseSimilarity } from 'posenet-similarity';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const posenetRef = useRef(false)
  const [distance, setDistance] = useState(0);
  const [maxDistance, setMaxDistance] = useState(distance)

  //  Load posenet
  const runPosenet = async () => {
    posenetRef.current = true
    const net = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    
    //
    setInterval(() => {
      detect(net);
    }, 100);
  };

  /**
   * @param {posenet.PoseNet} net 
   */
  const detect = async (net) => {
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
        strategy: 'cosineSimilarity'
      })
      setDistance(data)
      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
    }
  };

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;
    drawKeypoints(pose["keypoints"], 0.6, ctx);
    drawSkeleton(pose["keypoints"], 0.7, ctx);
  };

  useEffect(() => {
    if (!posenetRef.current){
      runPosenet();
    }
  })

  useEffect( () => {
    if(distance > maxDistance) {
      setMaxDistance(distance)
    }
  }, [distance, maxDistance])

  return (
    <div className="App">
      <header className="App-header">
          <Webcam
            ref={webcamRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zindex: 9,
              width: 640,
              height: 480,
            }}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zindex: 9,
              width: 640,
              height: 480,
            }}
          />
        <div className="image-score">
          <img ref={imgRef} crossOrigin="anonymous" src="https://ucarecdn.com/cae252d0-7c80-4523-834c-a0d3e053796f/-/preview/-/quality/smart/-/format/auto/" id="pose-match" style={{ width: "100%"}}/>
          <h1>{maxDistance.toFixed(2)}</h1>
        </div>
      </header>
    </div>
  );
}

export default App;
