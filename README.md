 Image processing

---------------------------------------------------------------------------------------------------------------------------------------------------
 1>. Develop the program to display grayscale image <br>
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
 output:  <br>
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
8>. Convert the original image to gray scale and then to binary <br>
import cv2  <br>
#read the image file   <br>
img=cv2.imread('plant1.jpg') <br>
cv2.imshow("RGB",img)  <br>
cv2.waitKey(0)  <br>
#Gray scale  <br>
img=cv2.imread('plant1.jpg',0)  <br>
cv2.imshow("Gray",img)  <br>
cv2.waitKey()  <br>
#Binary image  <br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)  <br>
cv2.imshow("Binary",bw_img)  <br>
cv2.waitKey(0)  <br> 
cv2.destroyAllWindows() <br>

![image](https://user-images.githubusercontent.com/97940146/178968844-94e316f3-c5cc-4875-82a3-5f20c0c6ba56.png)
![image](https://user-images.githubusercontent.com/97940146/178968979-1edc5543-f4ef-4253-9f10-fdf1f6a843ba.png)
![image](https://user-images.githubusercontent.com/97940146/178969056-e507642a-96b9-4bba-b239-0f94490652fc.png)

------------------------------------------------------------------------------------------------------------------------------------
9>.Resize the original image <br>
import cv2  <br>
img=cv2.imread('plant.jpg')  <br>
print('original image length width',img.shape)  <br>
cv2.imshow('original image',img)  <br>
cv2.waitKey(0)  <br>
#to show the resized image  <br>
imgresize=cv2.resize(img,(360,270))  <br>
cv2.imshow('Resized image',imgresize)  <br>
print('Resized image length width',imgresize.shape)  <br>
cv2.waitKey(0)  <br>
output: <br>
original image length width (1920, 1080, 3) <br>
Resized image length width (270, 360, 3) <br>

![image](https://user-images.githubusercontent.com/97940146/178971396-d01a62e5-0b7b-4e2d-aac2-23b2f1968649.png)
![image](https://user-images.githubusercontent.com/97940146/178971470-c65b5aae-ca5d-44eb-83b6-76e2852fea8e.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------
10>.<br>
from skimage import io <br>
import matplotlib.pyplot as pltg <br>
url='https://i.pinimg.com/originals/e6/7d/4e/e67d4e6ca4eb37a50aaed470e7abfb50.jpg' <br>
image=io.imread(url) <br>
plt.imshow(image) <br>
plt.show() <br>
![image](https://user-images.githubusercontent.com/97940146/180203594-90d2682b-d2b8-48af-be80-1d8982ad8863.png)

----------------------------------------------------------------------------------------------------------------------------------------------
11.Write a program to mask and blur the image<br>

import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread("leaf1.jpg")<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/181450717-46c4cab4-ec6e-45ab-9fb7-deee37320121.png)


hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br><br>
plt.imshow(mask,cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/181450878-e326839f-275b-4e45-9029-f8a113d9c927.png)
![image](https://user-images.githubusercontent.com/97940146/181450927-3c5fdf92-e68a-40c9-be4a-2c03223da407.png)


light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/181452859-22935c97-1913-43d5-a1c6-30a7795d9955.png)
![image](https://user-images.githubusercontent.com/97940146/181452890-9710d549-be6e-4038-8199-0f6423ecc91b.png)


final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/181453325-65ec8bd3-8ec6-4e84-9340-9e86224aaa53.png)
![image](https://user-images.githubusercontent.com/97940146/181453364-03d25436-2842-49cc-8813-7ac6d4e6c03e.png)


blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br><br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/181453427-17f6ddbe-fe54-47cd-a1d7-cf911731785e.png)

---------------------------------------------------------------------------------------------------------------------------------------
12>.Write a program to perform arithmatic operation on images<br>

import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('flower3.jpg')<br>
img2=cv2.imread('leaf1.jpg')<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>

![image](https://user-images.githubusercontent.com/97940146/183873587-ed487eaa-b493-4f42-b733-1e69ba0d8ba1.png)
![image](https://user-images.githubusercontent.com/97940146/183873701-86920cde-cb93-4d45-9fe7-ffbbcc5800f8.png)
![image](https://user-images.githubusercontent.com/97940146/183873766-9c5b9cbb-b10c-4e21-8dfd-b0a574f3a90b.png)
![image](https://user-images.githubusercontent.com/97940146/183873901-925fa805-9b8a-406e-aa62-dcaf3daf85ad.png)


-----------------------------------------------------------------------------------------------------------------------------------------------------
 13>. Develop the program to change the image to different color spaces<br>
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
14>.program to create an image using 2D array<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

![image](https://user-images.githubusercontent.com/97940146/178718467-3d023451-db44-4aa5-884d-08935213ceb3.png)
---------------------------------------------------------------------------------------------------------------------------------------------------------
15>.<br>
import cv2<br>
import matplotlib.pyplot as plt <br>
image1=cv2.imread('img2.jpg')<br>
image2=cv2.imread('img2.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940146/183869462-ba5d19b7-6738-4017-9280-096a08709503.png)


-------------------------------------------------------------------------------------------------------------------------------------------------------
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
23.<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt <br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread("img12.jpg")<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/97940146/179960063-24f8863f-3fc1-47a5-8ba5-6de375ee8695.png)

negative=255-pic<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/97940146/179960266-d7935183-6821-4a4c-934b-22340e480c2a.png)

import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('img12.jpg')<br>
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
     return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/97940146/179960437-8bb2b8ab-fa79-4ec9-893e-81b649b998fb.png)

import imageio<br>
import matplotlib.pyplot as plt  <br>
pic=imageio.imread('img12.jpg')<br>
gamma=2.2  <br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction);<br>
plt.axis('off'); <br>
![image](https://user-images.githubusercontent.com/97940146/179960523-57213a70-7e35-4be9-ac87-253c52c42647.png)

------------------------------------------------------------------------------------------------------------------------------------
24.<br>
#image sharpen  <br>
from PIL import Image  <br>
from PIL import ImageFilter  <br>
import matplotlib.pyplot as plt  <br>
#load the image  <br>
my_image=Image.open("img5.jpg")  <br>
#use the sharpen  <br>
sharp=my_image.filter(ImageFilter.SHARPEN) <br>
#save the image <br>
sharp.save("D:/image_sharpen.jpg") <br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940146/179960661-f76e670e-62c5-44d6-a127-dbfbb82705b8.png)

#image flip<br>
import matplotlib.pyplot as plt <br>
#load the image <br>
img=Image.open('img5.jpg')<br>
plt.imshow(img)<br>
plt.show() <br>
#use the flip function <br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT) <br>
#save image <br>
flip.save('D:/image_flip.jpg') <br>
plt.imshow(flip) <br>
plt.show() <br>
![image](https://user-images.githubusercontent.com/97940146/179960775-7b3e74e6-ea84-4cfd-ab06-f74dc3a56804.png)
![image](https://user-images.githubusercontent.com/97940146/179960802-f3824301-379c-45a4-b149-0b7a0d0bb115.png)

#Importing image class from PIL Module <br>
from PIL import Image <br>
import matplotlib.pyplot as plt <br>
#open a image in  RGB mode(size of the original image) <br>
im=Image.open('img5.jpg') <br>
#size of the image in pixels <br>
width,height=im.size <br>
im1=im.crop((280,100,800,600)) <br>
im1.show() <br>
plt.imshow(im1) <br><br>
plt.show() <br>
![image](https://user-images.githubusercontent.com/97940146/179960871-dd471008-eb99-4ea2-86dc-69dd393728bd.png)

---------------------------------------------------------------------------------
26>.Implement a program to perform various edge detection techniques<br><br>
a) Canny Edge detection<br><br>
#canny edge detection<br><br>
import cv2<br><br>
import numpy as np<br><br>
import matplotlib.pyplot as plt<br><br>
plt.style.use('seaborn')<br><br>

loaded_image = cv2.imread("shapes.jpg") <br><br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br><br>

gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br><br>

edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br><br>

plt.figure(figsize=(20,20))<br><br>
plt.subplot(1,3,1)<br><br>
plt.imshow(loaded_image,cmap="gray")<br><br>
plt.title("original Image")<br><br>
plt.axis("off")<br><br>
plt.subplot(1,3,2)<br><br>
plt.imshow(gray_image, cmap="gray")<br><br>
plt.axis("off")<br><br>
plt.title("GrayScale Image")<br><br>
plt.subplot(1,3,3) <br><br>
plt.imshow(edged_image,cmap="gray")<br><br>
plt.axis("off")<br><br>
plt.title("Canny Edge Detected Image")<br><br>
plt.show()<br><br>

![image](https://user-images.githubusercontent.com/97940146/187893093-28394195-b079-4cc6-9015-8951417dc553.png)<br>

#Laplacian and Sobel Edge detecting methods<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>

#Loading image<br>
#imge = cv2.imread("SanFrancisco.jpg',) <br>
imge = cv2.imread('shapes.jpg',)<br>

#converting to gray scale<br>
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)<br>
    # remove noise<br>
img = cv2.GaussianBlur (gray, (3,3),0)<br>

     # convolute with proper kernels<br>
laplacian= cv2.Laplacian(img,cv2.CV_64F) <br>
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #x<br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y<br>
plt.subplot(2,2,1), plt.imshow(img,cmap = 'gray')
plt.title("original"), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,2), plt.imshow(laplacian, cmap = 'gray') <br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx,cmap = 'gray')<br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray') 
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

![image](https://user-images.githubusercontent.com/97940146/187893795-246bf6c3-2515-4bde-a6b0-e1717db595c3.png)


               #Edge detection using Prewitt operator
               import cv2
               import numpy as np
               from matplotlib import pyplot as plt
               img =cv2.imread('shapes.jpg')
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
               img_gaussian = cv2.GaussianBlur (gray, (3,3),0)

               #prewitt
               kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]]) 
               kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
               img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx) 
               img_prewitty =cv2.filter2D(img_gaussian, -1, kernely)
               cv2.imshow("Original Image", img)
               cv2.imshow("Prewitt x", img_prewittx)
               cv2.imshow("Prewitt y", img_prewitty)
               cv2.imshow("Prewitt", img_prewittx + img_prewitty)
               cv2.waitKey()
               cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/97940146/187893404-c6728c4e-07a3-4a4b-861c-51f6af116f06.png)

