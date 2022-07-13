# Image processing

1>. Develop the program to display grayscale image <br>
 <br>
import cv2   <br>
img=cv2.imread('butterflypic.jpg',0)   <br>
cv2.imshow('image',img)   <br>
cv2.waitKey(0)   <br>
cv2.destroyAllWindows()   <br>
output:  <br>
![image](https://user-images.githubusercontent.com/97940146/178465559-246df71d-015a-42c2-8598-462a6982f52e.png)
-------------------------------------------------------------------------------------------------------------------------------------------------
2>.Develop a program to display image using matplotlib  <br>

import matplotlib.image as mping  <br>
import matplotlib.pyplot as plt  <br>
img=mping.imread('butterfly.jpg')  <br>
plt.imshow(img)  <br>

output: 
<matplotlib.image.AxesImage at 0x1ef8305fbe0>
![image](https://user-images.githubusercontent.com/97940146/178466748-befa1ba2-7a3a-464c-852a-692573a74175.png)
------------------------------------------------------------------------------------------------------------------------------------------------
3>.Develop a program to perfrorm linear transformation  (i)Rotation: <br>
from PIL import Image   <br>
img=Image.open('rose.jpg')   <br>
img=img.rotate(180)   <br>
img.show()   <br>
cv2.waitKey(0)   <br>
cv2.destoryAllWindows()  <br>
output: 
![image](https://user-images.githubusercontent.com/97940146/178471987-bf48b1e8-0b00-4480-bc2d-6511522b5a6c.png)

-------------------------------------------------------------------------------------------------------------------------------------------------
4>.Develop a program to convert color string to RGB color value  <br>
from PIL import ImageColor   <br>
img1=ImageColor.getrgb("yellow")   <br>
print(img1)   <br>
img2=ImageColor.getrgb("red")   <br>
print(img2)   <br>

output:  <br>

(255, 255, 0)  
(255, 0, 0)
----------------------------------------------------------------------------------------------------------------------------------------------------
5>.Write a program to create image using color  <br>
from PIL import Image  <br>
img=Image.new('RGB',(200,400),(255,255,0))  <br>
img.show()  <br>

-----------------------------------------------------------------------------------------------------------------------------------------------------
6.Program to visualize images using varoius volor spaces

import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('flower2.webp')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

![image](https://user-images.githubusercontent.com/97940146/178469962-e3d839b4-c1d0-4807-8486-b05eefd91eb7.png)
![image](https://user-images.githubusercontent.com/97940146/178470413-59acca4f-2058-48bc-b532-ddbde9aae67e.png)
![image](https://user-images.githubusercontent.com/97940146/178470716-ebfe263c-4608-4026-8520-7f6156f49491.png)

-------------------------------------------------------------------------------------------------------------------------------------
7.Program to display image attributes

from PIL import Image
image=Image.open('flower2.webp')
print("FileName:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)

OUTPUT:
FileName: flower2.webp
Format: WEBP
Mode: RGB
Size: (263, 300)
Width: 263
Height: 300
------------------------------------------------------------------------------------------------------------------------------------------------------


1> <br>
 import cv2 as c <br>
 import numpy as np <br>
 from PIL import Image <br>
 array=np.zeros([100,200,3],dtype=np.uint8) <br>
  array[:,:100]=[255,130,0] <br>
 array[:,100:]=[0,0,255] <br>
 img=Image.fromarray(array) <br>
 img.save('image1.png') <br>
 img.show() <br>
 c.waitKey(0) <br>
-------------------------------------------
2><br>
import cv2 <br>
img=cv2.imread("D:\plant.jpg") <br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) <br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) <br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB) <br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS) <br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) <br>
cv2.imshow("GRAY image",gray) <br>
cv2.imshow("HSV image",hsv) <br>
cv2.imshow("LAB image",lab) <br>
cv2.imshow("HLS image",hls) <br>
cv2.imshow("YUV image",yuv) <br>
cv2.waitKey(0) <br>
cv2.destroyAllWindows() <br>
OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940146/175283294-6c919061-6bc5-4028-9be5-9e4781eea7bb.png) <br>

16> <br>
#importing laibraries <br>
import cv2 <br>
import numpy as np <br>
image=cv2.imread('img4.jpg') <br>
cv2.imshow('original Image',image) <br>
cv2.waitKey(0) <br>
#Gaussian Blur <br>
Gaussian=cv2.GaussianBlur(image,(7,7),0) <br>
cv2.imshow('Gaussian Blurring',Gaussian) <br>
cv2.waitKey(0) <br>
#Median Blur <br>
median=cv2.medianBlur(image,5) <br>
cv2.imshow('Median Blurring',median) <br>
cv2.waitKey(0) <br>
#Bilateral Blur <br>
bilateral=cv2.bilateralFilter(image,9,75,75) <br>
cv2.imshow('Bilateral Blurring',bilateral) <br>
cv2.waitKey(0) <br>
cv2.destroyAllWindows() <br>
OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940146/176418191-b58c2d06-d645-469e-8387-4f3afe742021.png)
![image](https://user-images.githubusercontent.com/97940146/176418440-49580f3b-2874-4f2c-a097-7d87891ac43b.png)
![image](https://user-images.githubusercontent.com/97940146/176418576-fa96a001-8717-4506-be67-e5a4f099fbec.png)
![image](https://user-images.githubusercontent.com/97940146/176418704-2ad4d843-599b-4ea3-87f4-eb27c46e9b17.png)

17> <br>
from PIL import Image <br>
from PIL import ImageEnhance <br>
image=Image.open('img1.jpg')  <br>
image.show() <br>
enh_bri=ImageEnhance.Brightness (image) <br>
brightness=1.5 <br>
image_brightened = enh_bri.enhance (brightness) <br>
image_brightened.show() <br>
enh_col = ImageEnhance.Color(image) <br>
color = 1.5 <br>
image_colored = enh_col. enhance (color) <br>
image_colored.show() <br>
enh_con=ImageEnhance.Contrast (image) <br>
contrast = 1.5 <br>
image_contrasted = enh_con. enhance (contrast) <br>
image_contrasted.show() <br>
enh_sha =ImageEnhance. Sharpness (image) <br>
sharpness = 3.0 <br>
image_sharped = enh_sha. enhance (sharpness) <br>
image_sharped.show() <br>
![image](https://user-images.githubusercontent.com/97940146/176421167-54f456d5-e10b-4b10-bb07-00f1e9efd6e7.png)

18> <br>
![image](https://user-images.githubusercontent.com/97940146/178699921-2689536b-1fdc-416b-99cd-9cdbd4666421.png)

![image](https://user-images.githubusercontent.com/97940146/178699778-d7b710a9-dfb1-4157-9574-24d6ee081471.png)

![image](https://user-images.githubusercontent.com/97940146/178700352-44a4c10a-88a5-4ce5-9b2d-1cb5bbb7f633.png)
