import cv2
def shape_check(img):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    # list for storing names of shapes
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if cv2.contourArea(contour) < 1000 or cv2.contourArea(contour) > 10000:
            continue 
    
        #print(cv2.contourArea(contour))
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
        area = cv2.contourArea(contour)
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
    
        #print(len(approx))
        # putting shape name at center of each shape
        if len(approx) == 3:
            return "Triangle",area
    
        elif len(approx) == 4:
            return "Rectangle",area
    
        elif len(approx) == 5:
            return "Pentagon",area
     
        elif len(approx) == 12: #To fix
            return "X",area
        else:
            return "Circle",area