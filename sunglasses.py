import cv2

def overlay_img(img, img_over, img_over_x, img_over_y):
    img_h, img_w, img_c = img.shape
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
    sunGlass_img = cv2.imread("/Users/xuguoxiang/Desktop/手势识别/sunglass.jpg", cv2.IMREAD_UNCHANGED)
    height, width, channel = sunGlass_img.shape
    retval, image = capture.read()
    image = cv2.flip(image, 1)  # 屏幕反转
    # 加载识别人脸的级联分类器
    faceCascade = cv2.CascadeClassifier(
        "/Users/xuguoxiang/Library/Python/3.9/lib/python/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    # 识别出所有人脸
    faces = faceCascade.detectMultiScale(image, 1.15)
    for (x, y, w, h) in faces:
        gw = w
        gh = int(height * w / width)
        sunGlass_img = cv2.resize(sunGlass_img, (gw, gh))
        overlay_img(image, sunGlass_img, x, y + int(h * 8 / 30))
    cv2.imshow("Sunglasses faces", image)

    key = cv2.waitKey(1)
    if key == 32:
        break
capture.release()
cv2.destroyAllWindows()
