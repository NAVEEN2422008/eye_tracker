# Eye Detection Project

## Overview

This project implements real-time eye detection and blink detection using Python, OpenCV, and MediaPipe. The script analyzes video feed from a webcam to detect facial landmarks, calculate the Eye Aspect Ratio (EAR), and determine whether the user's eyes are open or closed. This can be used for applications like drowsiness detection, human-computer interaction, or accessibility tools.

## How It Works

### Eye Aspect Ratio (EAR)
The core algorithm uses the Eye Aspect Ratio, a metric proposed by Soukupová and Čech in their 2016 paper. The EAR is calculated as:

\[ EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 \times ||p1 - p4||} \]

Where p1-p6 are specific eye landmark points:
- p1, p4: Left and right eye corners (horizontal)
- p2, p3: Top eyelid points
- p5, p6: Bottom eyelid points

A lower EAR indicates closed eyes, while a higher EAR indicates open eyes.

#### Detailed EAR Calculation
In code, the `eye_aspect_ratio(eye)` function computes this for a single eye:
- `A = np.linalg.norm(eye[1] - eye[5])`: Euclidean distance between top-left and bottom-left eyelid points.
- `B = np.linalg.norm(eye[2] - eye[4])`: Euclidean distance between top-right and bottom-right eyelid points.
- `C = np.linalg.norm(eye[0] - eye[3])`: Euclidean distance between left and right eye corners.
- `ear = (A + B) / (2.0 * C)`: Average vertical distances divided by horizontal distance.

This ratio is scale-invariant and robust to slight head movements. Typical open-eye EAR values range from 0.2-0.4, while closed eyes drop below 0.2. The threshold of 0.25 is a common empirical value, but can be adjusted based on individual users or lighting conditions.

### MediaPipe Face Landmarker
The script uses Google's MediaPipe Face Landmarker, a pre-trained machine learning model that detects 478 facial landmarks in 3D space. The model processes video frames to identify face locations and landmark positions.

### Workflow
1. **Video Capture**: Captures frames from the webcam using OpenCV.
2. **Preprocessing**: Converts frames to RGB format for MediaPipe compatibility.
3. **Face Detection**: Uses MediaPipe to detect faces and extract landmark coordinates.
4. **Eye Landmark Extraction**: Isolates specific landmarks for left and right eyes.
5. **EAR Calculation**: Computes EAR for both eyes and averages them.
6. **Thresholding**: Compares average EAR to a threshold (0.25) to classify eye state.
7. **Visualization**: Draws eye contours and status text on the video feed.
8. **Display**: Shows the processed frame in real-time.

#### Detailed Workflow Breakdown
- **Frame Acquisition**: `cap.read()` retrieves a BGR frame from the webcam.
- **Color Conversion**: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` changes to RGB as MediaPipe expects SRGB format.
- **MediaPipe Processing**: `face_landmarker.detect_for_video()` runs inference on the frame, returning landmark data.
- **Coordinate Scaling**: Landmarks are normalized (0-1), so multiply by frame width/height to get pixel coordinates.
- **Eye Isolation**: Use predefined indices to extract 6 points per eye from the 478 total landmarks.
- **EAR Averaging**: Compute left and right EAR separately, then average for overall eye state.
- **State Logic**: If average EAR < 0.25, classify as closed; else open.
- **Rendering**: Use OpenCV to draw polygons, circles, and text overlays.
- **Loop Control**: Process at ~30 FPS, exit on 'q' keypress.

## MediaPipe Face Landmarker Details

### How It Works
The MediaPipe Face Landmarker is a sophisticated machine learning pipeline developed by Google for real-time facial analysis. It operates as a multi-stage process:

1. **Input Processing**: Accepts images or video frames, performing preprocessing like rotation, resizing, normalization, and color space conversion to prepare data for neural network inference.

2. **Face Detection**: Uses the BlazeFace short-range model, a lightweight convolutional neural network (CNN) optimized for mobile GPUs, to detect faces and provide initial bounding boxes with key facial landmarks.

3. **Landmark Localization**: The Face Mesh model, a more complex neural network, predicts 478 three-dimensional (3D) facial landmarks for each detected face. These landmarks cover the entire face, including eyes, nose, mouth, eyebrows, and contours.

4. **Expression Analysis**: The Blendshape prediction model analyzes the 3D landmarks to compute 52 "blendshape scores" – coefficients representing different facial expressions and muscle movements, enabling detailed surface reconstruction.

5. **Output Generation**: Produces 3D landmark coordinates, optional blendshape scores for expressions, and transformation matrices for applying virtual effects. Results are filtered by confidence scores to ensure accuracy.

The pipeline is optimized for edge computing, running efficiently on devices without constant internet connectivity, achieving real-time performance (~30 FPS) for video streams.

#### Model Specifications
- **Face Detector (BlazeFace)**: Input size 192x192, optimized for short-range detection, uses float16 precision for efficiency.
- **Face Mesh**: Input size 256x256, outputs 478 3D landmarks with x, y, z coordinates.
- **Blendshape Predictor**: Input size 1x146x2, predicts 52 expression coefficients.
- **Overall Bundle**: Packaged as a `.task` file (~20-30 MB), containing all models and metadata.
- **Performance**: Designed for real-time inference on CPUs/GPUs, with low latency for video processing.

### How It Was Created
- **Development**: Built by Google's AI team as part of the MediaPipe Solutions framework, an early-release platform for edge AI tasks.
- **Training Process**: The component models were trained on large, diverse datasets of facial images and videos using supervised learning. Neural networks learned to map input pixels to landmark coordinates by minimizing prediction errors on labeled training data.
- **Model Architectures**: Employs CNNs for feature extraction, with regression heads for coordinate prediction. The BlazeFace detector is specifically designed for short-range face detection, while the Face Mesh uses advanced mesh prediction techniques.
- **Packaging**: Trained models are bundled into downloadable `.task` files containing neural network weights, metadata, and configurations. These bundles are platform-agnostic, supporting Android, iOS, Web, and Python environments.
- **Availability**: Model bundles are hosted on Google's servers and available for download from MediaPipe documentation and GitHub repositories.

#### Training Details
- **Datasets**: Trained on proprietary Google datasets plus public face datasets, ensuring diversity in ethnicity, age, lighting, and poses.
- **Loss Functions**: Uses mean squared error for landmark regression, with additional losses for blendshape prediction.
- **Optimization**: Models are quantized to float16 for mobile deployment, balancing accuracy and speed.
- **Validation**: Tested on held-out datasets to ensure robustness across various conditions.

### What the Face Landmarker Looks Like
The Face Landmarker outputs a comprehensive 3D mesh of the face with 478 landmark points. These points form a dense mapping of facial features:

- **Eyes**: ~20 points per eye, including corners, eyelids, and iris contours.
- **Nose**: Points along the bridge, tip, and nostrils.
- **Mouth**: Inner and outer lip contours, corners, and teeth line.
- **Eyebrows**: Arches and edges.
- **Face Contour**: Jawline, cheeks, forehead, and ears.
- **Other**: Chin, temples, and fine details.

Visual representation shows a wireframe mesh overlay on the face, with points connected to form triangles representing the facial surface. For a detailed view, see MediaPipe's [full-size landmark image](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png).

In your code, the `face_landmarker.task` file is this pre-trained bundle loaded into memory.

### How It's Used in the Code
In `eye_detection.py`, the Face Landmarker is integrated as follows:

1. **Initialization**:
   ```python
   options = FaceLandmarkerOptions(
       base_options=BaseOptions(model_asset_path='face_landmarker.task'),
       running_mode=VisionRunningMode.VIDEO
   )
   face_landmarker = FaceLandmarker.create_from_options(options)
   ```
   - Loads the model from the `.task` file.
   - Configures for video mode to handle continuous frames.

2. **Frame Processing**:
   ```python
   results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
   ```
   - Processes each RGB frame to detect faces and landmarks.
   - Returns results with `face_landmarks` containing 478 points per face.

3. **Landmark Extraction**:
   - Scales normalized coordinates (0-1) to pixel coordinates using frame dimensions.
   - Selects specific indices for eyes: `LEFT_EYE = [362, 385, 387, 263, 373, 380]` and `RIGHT_EYE = [33, 160, 158, 133, 153, 144]`.

4. **Integration with EAR**:
   - Passes extracted eye points to `eye_aspect_ratio()` for blink detection.
   - Enables real-time analysis by processing ~30 frames per second.

The model's efficiency allows for low-latency detection, making it ideal for interactive applications. Confidence thresholds can be adjusted for different environments, and the 3D nature of landmarks provides depth information for advanced use cases.

## Files

- `eye_detection.py`: Main script containing the eye detection logic.
- `face_landmarker.task`: Pre-trained MediaPipe model file (download from MediaPipe repository).

## Dependencies

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)

## Installation

1. Clone or download the project files.
2. Install dependencies: `pip install opencv-python mediapipe numpy`
3. Download the Face Landmarker model from [MediaPipe's model repository](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place `face_landmarker.task` in the project directory.

## Usage

Run the script:

```bash
python eye_detection.py
```

- A window will open showing the video feed with eye landmarks highlighted.
- Green indicates open eyes, red indicates closed eyes.
- Press 'q' to quit.

## Configuration

- **EAR_THRESHOLD**: Adjust the threshold (default 0.25) based on lighting conditions and user preferences. Lower values increase sensitivity to blinks.
- **Eye Landmarks**: The landmark indices are based on MediaPipe's face mesh. Modify `LEFT_EYE` and `RIGHT_EYE` arrays if needed for different face orientations.
- **MediaPipe Options**: Customize `FaceLandmarkerOptions` for num_faces (default 1), confidence thresholds, or output settings like blendshapes.
- **Video Settings**: Change webcam device with `cv2.VideoCapture(1)` for secondary camera, or adjust frame rate.

## Troubleshooting

- **No Face Detected**: Ensure good lighting and frontal face position. Increase `min_face_detection_confidence` if needed.
- **Inaccurate EAR**: Calibrate threshold per user; print EAR values for tuning.
- **Performance Issues**: Reduce frame processing if CPU is overloaded; MediaPipe is optimized but may lag on low-end hardware.
- **Model Loading Errors**: Verify `face_landmarker.task` path and integrity; download latest version from MediaPipe.
- **OpenCV Errors**: Check webcam permissions and drivers.

## Potential Improvements

- **Multi-Face Support**: Set `num_faces > 1` to detect multiple people.
- **Head Pose Estimation**: Use additional landmarks for 3D head orientation.
- **Advanced Blink Detection**: Implement temporal filtering for more robust blink counting.
- **Integration**: Combine with drowsiness alerts or gaze tracking.
- **Custom Training**: Fine-tune the model on specific datasets using MediaPipe Model Maker.

## Limitations

- Requires good lighting for accurate detection.
- Works best with frontal face orientation.
- Performance depends on hardware (optimized for real-time on modern CPUs/GPUs).
- May struggle with extreme angles, occlusions, or diverse demographics if not in training data.

## References

- Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. 21st Computer Vision Winter Workshop.
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- BlazeFace Paper: [Short-Range Face Detection](https://arxiv.org/abs/1907.05047)
- Face Mesh: [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_mesh)

## License

This project is for educational purposes. Check MediaPipe and OpenCV licenses for commercial use.