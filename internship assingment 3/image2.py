from PIL import Image 
import cv2
import numpy as np
  
# Opens a image in RGB mode 
im = Image.open('try2.jpeg.jpg')
img = cv2.imread('try2.jpeg.jpg')
print(img.shape)

  
# Setting the points for cropped image 
left = 360
top = 330
right = 600
bottom = 420
  
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
  
# Shows the image in image viewer 
im1.save('Data\main2-1.jpg')
Image.open('Data\main2-1.jpg')



#code for segmentation
import cv2
from PIL import Image 
from pytesseract import image_to_string 

#img = Image.open("/Users/KHUSH/Downloads/segmented.jpg") 
img = cv2.imread('Data\main2-1.jpg')
mser = cv2.MSER_create()

#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
img = img[5:-5,5:-5,:]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()

regions = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
cv2.polylines(vis, hulls, 1, (0,255,0)) 

cv2.imwrite("Data\main2-2.jpg",img)
cv2.namedWindow('img', 0)
#print(image_to_string(Image.open('/Users/KHUSH/Downloads/segmented2.jpg'),lang='eng'))
cv2.imshow('img', vis)
while(cv2.waitKey()!=ord('q')):
       continue
cv2.destroyAllWindows()



try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
img = Image.open('try2.jpeg.jpg') 

def ocr_core(img):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(img))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

print(ocr_core('try2.jpeg.jpg'))