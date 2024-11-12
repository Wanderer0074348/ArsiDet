#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PATH_ROOT = "."
# PATH_ROOT = "D:\\SiddharathStuff\\"

PATH = f"/home/prakashhd/Documents/SIC/archive/datasets/sign.yaml"
PATH_MODEL = f"/home/prakashhd/Documents/SIC/trained_model.h5"
API_KEY = "oJiV8nDn6XoEN1n94dLlxfth9"


# In[2]:


get_ipython().system('pip3 install ultralytics')
get_ipython().system('pip3 install comet-ml')


# In[3]:


import comet_ml
exp = comet_ml.start(api_key=API_KEY)

from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


# In[4]:


# Purpose: to enable GPU

get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')

import torch
import torchvision
print(torch.cuda.is_available())
torch.cuda.set_device(0)



# In[5]:


# model = YOLO('yolov8n.yaml')
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml').load('yolov8n.pt').to("cuda")


# In[6]:


history = model.train(data=PATH, epochs=100, imgsz=256,
                    patience = 100, batch = 128,
                    project ="ASL", optimizer = 'Adam', momentum = 0.9,
                    cos_lr=True ,seed = 42, plots = True , close_mosaic = 0, lr0 = 0.001)
exp.end()


# In[7]:


model.save('ArabicSignLanguage.pt')

