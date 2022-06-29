# IP
1> 
 import cv2 as c
 import numpy as np
 from PIL import Image
 array=np.zeros([100,200,3],dtype=np.uint8)
  array[:,:100]=[255,130,0]
 array[:,100:]=[0,0,255]
 img=Image.fromarray(array)
 img.save('image1.png')
 img.show()
 c.waitKey(0)
-------------------------------------------
2>
import cv2 
img=cv2.imread("D:\plant.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT:
![image](https://user-images.githubusercontent.com/97940146/175283294-6c919061-6bc5-4028-9be5-9e4781eea7bb.png)

16>
#importing laibraries
import cv2 
import numpy as np
image=cv2.imread('img4.jpg')
cv2.imshow('original Image',image)
cv2.waitKey(0)
#Gaussian Blur
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)
#Median Blur
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)
#Bilateral Blur
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT:
![image](https://user-images.githubusercontent.com/97940146/176418191-b58c2d06-d645-469e-8387-4f3afe742021.png)
