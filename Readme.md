# Vis-DiceX<sup>2</sup>
> Vis-Dice<sup>2</sup> is a computer vision web-based tool build using streamlit to perform three functions: Inference, Training and Evaluation.

## Requirements
- **Python**: 3.10
- **CUDA Toolkit**: Compatible with Torch
- **Torch**: Required for deep learning operations

## Model Weights
You can install the YOLO model weights Here: [Download Link](https://drive.google.com/drive/folders/14hduF6_zP0yVD9t2IzsBDQ6UFfifeM7M?usp=sharing). Transfer the weights in the ``models/`` folder.

## Execution Guide

### Step 1: Create the Environment
Run the following commands to create and activate a new Conda environment:

```bash
conda create -n visdicex2 python=3.10 -y
conda activate visdicex2
```
### Install PyTorch 
Install PyTorch and CUDA libraries (CUDA version: 12.4) using Conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Install Dependencies
```
cd VisDiceX2
pip install -r requirements
```
### Notes
You can ensure torch is installed with cuda (inference/training on GPU) by executing the following code
```bash
python utils/check_torch_cuda.py
```
### 

