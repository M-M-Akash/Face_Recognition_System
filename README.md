# Real Time Multiple Cameras Face Recognition System 
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
If you do have the gpu support you need to install the gpu version. You can find the command here selecting your cuda version.
 [paddle paddle ](https://www.paddlepaddle.org.cn/). 

## 3. Install Wheel Package

Install the Wheel package using pip:

```bash
pip install wheel
```

## 4. Install InsightFacePaddle

We will be using the InsightFacePaddle package for face recognition. Clone this repository:

```bash
cd insight-face-paddle
```

Build the wheel package and install it using the following commands:

```bash
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

For face detection, we'll use the PyramidBox Lite Server module from PaddleHub. Install it using the following command:

```bash
hub install pyramidbox_lite_mobile
```

This will create a `.paddlehub` directory in your home folder and store the module there.

---

You have now completed the setup process for face recognition using PaddlePaddle. Follow the next steps specific to your application to continue. Feel free to refer to the provided links for further documentation on each component.
