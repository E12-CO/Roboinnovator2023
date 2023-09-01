import numpy as np
import cv2
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('images/raw2.png')

def wrap_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000 :
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                return approx
def main():
    image = cv2.imread('images/raw2.png')
    approx = wrap_image(image)
    print(list(approx[0][0]))
    print(list(approx[1][0]))
    print(list(approx[2][0]))
    print(list(approx[3][0]))

    # print(approx)
    pts1 = np.float32([list(approx[1][0]),list(approx[0][0]),list(approx[2][0]),list(approx[3][0])])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image,M,(300,300))
    plt.subplot(121),plt.imshow(image),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

if __name__ == '__main__':
    main()
