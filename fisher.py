import cv2
import numpy as np


def cutting_to_face(image):
    image = cv2.medianBlur(image, 1)  # 利用中值滤波器去除噪点
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Binary", binary)
    # 获取二值化图像中的轮廓及轮廓层次
    contours, hietatchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    #cv2.imshow("contours", image)

    # 加载识别人脸的级联分类器
    faceCascade = cv2.CascadeClassifier(
        "/Users/xuguoxiang/Library/Python/3.9/lib/python/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    # 识别出所有人脸
    faces = faceCascade.detectMultiScale(image, 1.15)
    print(faces)
    if len(faces) > 1:
        print('人脸识别失败')
    else:
        pass
    try:
        dstImg = image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
    except IndexError as e:
        print(e)
    return dstImg

# 按比例缩小图片尺寸
from PIL import Image
def suofang(im):
    im = cutting_to_face(im)
    '''cv2.imshow("dstImg", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    dim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    '''cv2.imshow("dImg", dim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    '''print(im)
    imcopy = np.copy(im)
    print(imcopy)'''
    # (x, y) = im.size  # 读取图片尺寸（像素）
    x_s = 300  # 定义缩小后的标准宽度
    # y_s = int(y * x_s / x)  # 基于标准宽度计算缩小后的高度
    y_s = 300  # 基于标准宽度计算缩小后的高度
    dim = cv2.resize(dim,(x_s, y_s))  # 改变尺寸，保持图片高品质
    # out.save('pictures_new3.png')
    print(dim)
    '''cv2.imshow("dstImg", dim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return dim


photos = list()
labels = list()
j_index = 0
for i in range(5):
    im = cv2.imread('Xu_'+str(i+1)+'.jpg')
    im = suofang(im)
    photos.append(im)
    labels.append(j_index)
j_index += 1
for i in range(2):
    im = cv2.imread('yuan_'+str(i+1)+'.jpg')
    im = suofang(im)
    photos.append(im)
    labels.append(j_index)

recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.train(photos,np.array(labels))

capture = cv2.VideoCapture(0)
label, confidence = -1,-1
n_index = 0
while(capture.isOpened()):
    retval,image = capture.read()
    image = cv2.flip(image, 1)  # 屏幕反转
    cv2.imshow("Video", image)

    try:
        image = suofang(image)
        label, confidence = recognizer.predict(image)
    except UnboundLocalError as e:
        print(e)
    except cv2.error as ee:
        print(ee)
    print('confidence = ' + str(confidence))
    if confidence <= 250 and confidence >= 0:
        if label == 0:
            print('It is Xu')
            n_index += 1
        if label == 1:
            print('It is Yuan')
            n_index += 1
    else:
        print('Unknow')

    if n_index >= 20:
        print('test successfully!')
        break

    key = cv2.waitKey(1)
    '''
    key = cv2.waitKey(1)
    if key == 32:
        image = suofang(image)
        label, confidence = recognizer.predict(image)
        break
print('confidence = '+ str(confidence))
if confidence <= 5000 and confidence >= 0:
    if label == 0:
        print('It is Xu')
    if label == 1:
        print('It is Yuan')
else:
    print('Unknow')
    '''
capture.release()
cv2.destroyAllWindows()