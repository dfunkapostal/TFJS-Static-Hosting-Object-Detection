import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import * as tfd from "@tensorflow/tfjs-data";
import Webcam from "react-webcam";
import { drawRect } from "./utilities";
import "./styles.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.webcamRef = React.createRef();
    this.canvasRef = React.createRef();
    this.state = {
      imageSrc: null
    };
  }

  async componentDidMount() {
    this.model = await tf.loadGraphModel(
      "https://raw.githubusercontent.com/hugozanini/TFJS-object-detection/master/models/kangaroo-detector/model.json"
    );

    this.metadata = await tfd.util.fetch(
      "https://raw.githubusercontent.com/dfunkapostal/TFJS-Static-Hosting-Object-Detection/main/models/Dwarf-Shrimp-DetectorV2/metadata.json"
    );

    this.colors = await tfd.util.fetch(
      "https://raw.githubusercontent.com/dfunkapostal/TFJS-Static-Hosting-Object-Detection/main/models/Dwarf-Shrimp-DetectorV2/colors.json"
    );

    setInterval(() => {
      this.detectObjects();
    }, 10);
  }

  handleImageUpload = (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = () => {
      const canvas = this.canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      this.detectObjects();
    };
  };
};


  detectObjects = async () => {
    const video = this.webcamRef.current.video;
    const { videoWidth, videoHeight } = video;

    const tfImg = tf.browser.fromPixels(video);
    const smallImg = tf.image.resizeBilinear(tfImg, [416, 416]);
    const resized = tf.cast(smallImg, "float32").div(tf.scalar(255));
    const input = tf.expandDims(resized, 0);

    const { boxes, scores, classes } = await this.model.executeAsync({
      image_tensor: input,
    });

    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const threshold = 0.5;

    const boxesData = await boxes.data();
    const scoresData = await scores.data();
    const classesData = await classes.data();

    for (let i = 0; i < scoresData.length; i++) {
      const score = scoresData[i];

      if (score > threshold) {
        const [y, x, h, w] = boxesData.slice(i * 4, (i + 1) * 4);

        const minY = Math.max(0, (y * videoHeight) / 416);
        const minX = Math.max(0, (x * videoWidth) / 416);
        const maxY = Math.min(videoHeight, (h * videoHeight) / 416);
        const maxX = Math.min(videoWidth, (w * videoWidth) / 416);

        const bbox = [minX, minY, maxX - minX, maxY - minY];

        const classId = classesData[i];
        const className = this.metadata[classId]["name"];
        const color = this.colors[className];

        drawRect(bbox, ctx, color);

        const text = `${className} ${Math.round(score * 100)}%`;
        const textX = minX > 10 ? minX - 5 : minX + 5;
        const textY = minY > 10 ? minY - 5 : minY + 20;

        ctx.font = "14px Arial";
        ctx.fillStyle = "#fff";
        ctx.fillText(text, textX, textY);
      }
    }

    tf.dispose([tfImg, smallImg, resized, input, boxes, scores, classes]);
    requestAnimationFrame(() => {
      this.detectObjects();
    });
  };

  render() {
    return (
      <div>
        <h1>Real-Time Object Detection: Dwarf Shrimp</h1>
        <h3>MobileNetV2</h3>
        <div>
          <label htmlFor="imageUpload">Upload an image:</label>
          <input type="file" id="imageUpload" accept="image/*" onChange={this.handleImageUpload} />
        </div>
        {this.state.imageSrc && (
          <div>
            <h4>Uploaded Image:</h4>
            <img src={this.state.imageSrc} alt="Uploaded Image" width="416" height="416" />
          </div>
        )}
        <video
          style={{height: '416px', width: "416px"}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.webcamRef}
          width="416"
          height="416"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="416"
          height="416"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
