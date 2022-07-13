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
img=Image.new('RGB',(300,400),(0,255,0))  <br>
img.show()  <br>
 
 ![image](https://user-images.githubusercontent.com/97940146/178716729-ec8b09be-9cf2-40ec-b3e4-921c145f214c.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------
6.Program to visualize images using varoius volor spaces

import cv2  <br>
import matplotlib.pyplot as plt  <br>
import numpy as np  <br>
img=cv2.imread('flower2.webp')  <br>
plt.imshow(img)  <br>
plt.show()  <br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) <br>
plt.imshow(img)  <br>
plt.show()  <br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)  <br>
plt.imshow(img)  <br>
plt.show()  <br>

![image](https://user-images.githubusercontent.com/97940146/178469962-e3d839b4-c1d0-4807-8486-b05eefd91eb7.png)
![image](https://user-images.githubusercontent.com/97940146/178470413-59acca4f-2058-48bc-b532-ddbde9aae67e.png)
![image](https://user-images.githubusercontent.com/97940146/178470716-ebfe263c-4608-4026-8520-7f6156f49491.png)

-------------------------------------------------------------------------------------------------------------------------------------
7.Program to display image attributes  <br>

from PIL import Image  <br>
image=Image.open('flower2.webp')  <br>
print("FileName:",image.filename)  <br>
print("Format:",image.format)  <br>
print("Mode:",image.mode)  <br>
print("Size:",image.size)  <br>
print("Width:",image.width)  <br>
print("Height:",image.height)  <br>

OUTPUT: <br>
FileName: flower2.webp  <br>
Format: WEBP  <br>
Mode: RGB  <br>
Size: (263, 300)  <br>
Width: 263  <br>
Height: 300  <br>

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
 
-----------------------------------------------------------------------------------------------------------------------------------------------------
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

------------------------------------------------------------------------------------------------------------------------------------------------------------
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

------------------------------------------------------------------------------------------------------------------------------------------------------
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

-----------------------------------------------------------------------------------------------------------------------------------------------------------
 18>.<br>
import cv2  <br>
import numpy as np  <br>
from matplotlib import pyplot as plt  <br>
from PIL import Image,ImageEnhance  <br>
img=cv2.imread('img3.jpg',0) <br>
ax=plt.subplots(figsize=(20,10))  <br>
kernel=np.ones((5,5),np.uint8)  <br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)  <br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)  <br>
erosion=cv2.erode(img,kernel,iterations=1)  <br>
dilation=cv2.dilate(img,kernel,iterations=1) <br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel) <br>
plt.subplot(151) <br>
plt.imshow(opening)  <br>
plt.subplot(152)  <br>
plt.imshow(closing)  <br>
plt.subplot(153)  <br>
plt.imshow(erosion) <br>
plt.subplot(154)  <br>
plt.imshow(dilation)  <br>
plt.subplot(155)  <br>
plt.imshow(gradient) <br>
cv2.waitKey(0) <br>

![image](https://user-images.githubusercontent.com/97940146/178708500-3c4036ab-08ff-47c6-941d-e36098335714.png)


------------------------------------------------------------------------------------------------------------------------------------------------
19>.<br
        
![image](https://user-images.githubusercontent.com/97940146/178699921-2689536b-1fdc-416b-99cd-9cdbd4666421.png)

![image](https://user-images.githubusercontent.com/97940146/178699778-d7b710a9-dfb1-4157-9574-24d6ee081471.png)

![image](https://user-images.githubusercontent.com/97940146/178700352-44a4c10a-88a5-4ce5-9b2d-1cb5bbb7f633.png)

20>.<br>
import cv2  <br>
import numpy as np  <br>
from matplotlib import pyplot as plt  <br>
image=cv2.imread('img5.jpg',0)  <br>
x,y=image.shape  <br>
z=np.zeros((x,y))  <br>
for i in range(0,x):  <br>
    for j in range(0,y):  <br>
        if(image[i][j]>50 and image[i][j]<150):  <br>
            z[i][j]=255  <br>
        else:  <br>
            z[i][j]=image[i][j]  <br>
equ=np.hstack((image,z))  <br>
plt.title('Graylevel slicing with background') <br>
plt.imshow(equ,'gray')  <br>
plt.show()  <br>

![image](https://user-images.githubusercontent.com/97940146/178705790-ed2298ef-7564-4f0c-959d-d124121c4307.png)

-----------------------------------------------------------------------------------------------------------------------------------------------
21>. <br>
import cv2 <br>
import numpy as np <br>
from matplotlib import pyplot as plt <br>
image=cv2.imread('img8.jpg',0) <br>
x,y=image.shape  <br>
z=np.zeros((x,y))  <br>
for i in range(0,x):   <br>
    for j in range(0,y):  <br>
        if(image[i][j]>50 and image[i][j]<150):  <br>
            z[i][j]=255  <br>
        else:  <br>
            z[i][j]=0  <br>
equ=np.hstack((image,z))  <br>
plt.title('Graylevel slicing with background')  <br>
plt.imshow(equ,'gray')  <br>
plt.show()  <br>

![image](https://user-images.githubusercontent.com/97940146/178706595-cb64cb2d-8ef0-4b74-b76e-990883e3fef5.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------
