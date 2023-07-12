# Real Time Multiple Cameras Face Recognition System 

## Development Pipeline

The development pipeline for the face recognition system consists of several key steps, including model selection, data preprocessing, training, and deployment. The following is an overview of the pipeline:

1. **Model Selection:**
   - Initially, the system utilized the `insight-face-paddle` library developed by PaddlePaddle for face recognition.
   - However, due to computational limitations on a CPU-based system, it was necessary to optimize the face detection model.

2. **Face Detection Model Optimization:**
   - The original face detection model, BlazeFace, was computationally expensive for continuous frame processing.
   - As an alternative, the PyramidBox model was chosen as it demonstrated robustness against interferences and was optimized for mobile devices.
   - The lightweight version of the PyramidBox model was preferred to ensure efficient operation on embedded systems and mobile devices.

3. **Integration of Face Detection and Recognition:**
   - The face detection model was integrated with the existing face recognition system from `insight-face-paddle`.
   - When processing frames from a video stream, the face detection model detects faces and extracts face crop images.
   - These face crop images are then passed through the face recognition model, MobileFace, to generate face embeddings.

4. **Similarity Measurement:**
   - The face embeddings obtained from the MobileFace face recognition model are used to compute the cosine similarity between the camera feed faces and provided image faces.
   - This similarity measurement helps determine the degree of resemblance between faces, enabling face recognition and identification.

5. **Multithreading for Model Inference:**
   - To fully utilize available system resources, multithreading techniques were employed to handle the model inferencing operation.
   - Multithreading ensures that both face detection and face recognition models can make predictions concurrently, optimizing system performance.

6. **GPU Support:**
   - Additionally, a script for GPU support was developed, enabling the system to leverage GPU acceleration if available.
   - The GPU support script enhances the overall processing speed and allows for more efficient utilization of computational resources.


# Setup Process

This guide outlines the steps to set up the required Python environment and install the necessary packages for face recognition using PaddlePaddle. Follow the instructions below to get started:

## 1. Create Python Virtual Environment

To begin, create a Python virtual environment using Python 3.8. Make sure you have pip version 20.0.2 installed. Run the following commands:

```bash
python3.8 -m venv myenv        # Replace `myenv` with your preferred environment name
source myenv/bin/activate    # Activate the virtual environment
```

## 2. Install PaddlePaddle Framework

Next, we need to install the PaddlePaddle framework. We'll be using version 2.4.2 with cpu support. Use the following command:

```bash
python -m pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If you do have the gpu support you need to install the gpu version. You can find the command here selecting your cuda version
 [here](https://www.paddlepaddle.org.cn/).

## 3. Install Wheel Package

Install the Wheel package using pip:

```bash
pip install wheel
```

## 4. Install InsightFacePaddle

We will be using the InsightFacePaddle package for face recognition. Build the wheel package from this repository insight-face-paddle directory.

Install it using the following commands:

```bash
cd insight-face-paddle
python setup.py bdist_wheel
pip install dist/*
```


## 5. Install PaddlePaddle Hub

PaddlePaddle Hub is a powerful toolkit for pretrained AI models based on paddle paddle framework. Install it using the following command:

```bash
pip install paddlehub==2.1.0
```

To verify the installation, run the following code snippet:

```python
import paddlehub

paddlehub.server_check()
```

## 6. Install Face Detection Module

For face detection, we'll use the PyramidBox Lite Mobile module from PaddleHub. Install it using the following command:

```bash
hub install pyramidbox_lite_mobile
```

You have now completed the setup process for face recognition using PaddlePaddle. Follow the next steps specific to your application to continue. Feel free to refer to the provided links for further documentation on each component.
