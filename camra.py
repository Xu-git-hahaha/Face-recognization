import cv2

def overlay_img(img, img_over, img_over_x, img_over_y):
    img_w, img_h, img_c = img.shape
    img_over_h, img_over_w, img_over_c = img_over.shape
    if img_over_c == 3:
        img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2BGRA)
    for w in range(0,img_over_w):
        for h in range(0, img_over_h):
            if img_over[h,w,3] !=0:
                for c in range(0,3):
                    x = img_over_x + w
                    y = img_over_y + h
                    if x >= img_w or y >= img_h:
                        break
                    img[y, x, c] = img_over[h, w, c]
    return img


capture = cv2.VideoCapture(0)
while(capture.isOpened()):
    retval,image = capture.read()
    image = cv2.flip(image, 1)  # 屏幕反转
    ###cv2.imshow("Video", image)

    image = cv2.medianBlur(image,1)  # 利用中值滤波器去除噪点
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", binary)
    # 获取二值化图像中的轮廓及轮廓层次
    contours, hietatchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,contours,-1,(255,0,0),2)
    cv2.imshow("contours", image)

    retval, image = capture.read()
    image = cv2.flip(image, 1)  # 屏幕反转
    # 加载识别人脸的级联分类器
    faceCascade = cv2.CascadeClassifier("/Users/xuguoxiang/Library/Python/3.9/lib/python/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
    # 识别出所有人脸
    faces = faceCascade.detectMultiScale(image, 1.15)
    for (x,y,w,h) in faces:
        # 在图像中人脸的位置绘制方框
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 5)
    cv2.imshow("Faces", image)

    key = cv2.waitKey(1)
    if key == 32:
        break
capture.release()
cv2.destroyAllWindows()
