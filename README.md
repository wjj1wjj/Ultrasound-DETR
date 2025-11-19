# Ultrasound-DETR
Ultrasound-DETR: A Decoupled and Hybrid DETR Network for Ultrasound Nodule Detection

This is the code repository for Ultrasound-DETR.   

# Installation and Usage  
1.Clone this repo  
```
git clone https://github.com/wjj1wjj/Ultrasound-DETR.git  
cd Ultrasound-DETR
```
2.Install Pytorch and torchvision  
```
# an example:
conda create -n Ultrasound-DETR python=3.8
conda activate Ultrasound-DETR
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
3.Install other needed packages
```
pip install -r requirements.txt
```
# Data
Please organize your dataset as following: 
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
# Run
```
python Ultrasound-DETR/main.py
```
# Contributing  
Feel free to submit issues or pull requests to contribute to the project and improve it.
# License  
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.











