# import cv2
#
#
# img = cv2.imread('images/image.jpg')
# img1 = cv2.imread('images/image1.jpg')
#
# img1 = cv2.resize(img1, (800, 500))
# img = cv2.resize(img, (800, 500))
# # result = cv2.addWeighted(img, 0.5, img1, 0.5, -100) # gamma is light/dark
#
#
#
#
# # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
# # img = cv2.GaussianBlur(img, (9, 9), 0)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # img = cv2.Canny(img, 90, 90)
# cv2.imshow('1', result)
#
# # print(img.shape)
#
# cv2.waitKey(0)

# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# while (True):
#
#     ret, frame = cap.read()
#     # frame = cv2.flip(frame, -1)  # Flip camera vertically
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # cv2.imshow('frame', frame)
#
#     cv2.imshow('gray', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# cap.set(3, 500)
# cap.set(4, 500)
#
# while True:
#     success, img = cap.read()
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.Canny(img, 45, 45)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     cv2.imshow('1', img)

# import required libraries
# import OpenCV library
import cv2


# import matplotlib library
# importing time library for speed comparisons of both classifiers
# %matplotlib inline

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# load test iamge
test1 = cv2.imread('images/image.jpg')
# convert the test image to gray image as opencv face detector expects gray images
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

haar_face_cascade = cv2.CascadeClassifier(
    'C:/Users/maser/Desktop/Проекты/Python/pythonProject2/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

print(help(haar_face_cascade.detectMultiScale))
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

# print the number of faces found
print('Faces found: ', len(faces))

cv2.imshow('Test Imag', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
