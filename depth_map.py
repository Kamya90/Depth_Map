#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread(r"C:\Users\blankblack\Desktop\left.png",0)#left image
imgR = cv.imread(r"C:\Users\blankblack\Desktop\right.png",0)#right image
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()


# In[ ]:





# In[ ]:




