import cv2 as cv
import time
import pytesseract as tess
from PIL import Image


def recognize_text(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值化
    # cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # 结构元素
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 开操作
    # cv.imshow("bin1", bin1)

    kerne2 = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # 结构元素
    open_out = cv.morphologyEx(bin1, cv.MORPH_OPEN, kerne2)  # 开操作
    cv.imshow("open_out", open_out)

    cv.bitwise_not(open_out, open_out)
    textImage = Image.fromarray(open_out)
    text = tess.image_to_string(textImage)
    print("识别结果： %s" % text)


if __name__ == "__main__":
    src = cv.imread("C:/Users/aa092/OneDrive/桌面/破解驗證碼python/python_captcha/6547.jpg")
    start = time.time()

    cv.imshow("input image", src)
    recognize_text(src)

    end = time.time()
    print('Running time: %s Seconds' % (end-start))

    cv.waitKey(0)
    cv.destroyAllWindows()