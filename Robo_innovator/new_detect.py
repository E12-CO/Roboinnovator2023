import cv2
import numpy as np
def find_largest_rectangle(img,cThr=[100,100],showCanny=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny:cv2.imshow('Canny',imgThre)
    contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter based on the contour area (adjust this threshold as needed)
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Only consider contours with four vertices (approximating rectangles)
                filtered_contours.append(approx)

    # Step 5: Find the largest rectangle among the filtered contours
    if len(filtered_contours) == 0 :
        return img
    largest_rectangle = max(filtered_contours, key=cv2.contourArea)
    cv2.drawContours(largest_rectangle, [largest_rectangle], 0, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(largest_rectangle)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    return img[y:y+h, x:x+w]

def main():
    cap = cv2.VideoCapture(0)
    while True:
        success,frame = cap.read()
        img = find_largest_rectangle(frame)
        cv2.imshow("windows",img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()
