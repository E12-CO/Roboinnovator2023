import cv2
import numpy as np
import pytesseract

def binarycheck(img):
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 1, maxRadius = 40)
    
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        binarymap = dict()

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            halfr = int(r/2)
            dumimg = img[b-halfr:b+halfr, a-halfr:a+halfr]

            threshold = 0.8 #threshold to determine if circle is white/black
            dumcnt = 0

            for h in range(dumimg.shape[0]):
                for w in range(dumimg.shape[1]):
                    if int(dumimg[h,w][0]) + int(dumimg[h,w][1])+ int(dumimg[h,w][2]) > 750:  #if pixel is white
                        dumcnt+=1
            res = float((dumcnt)/(dumimg.shape[0]*dumimg.shape[1])) 
            if res > threshold:
                #if result is higher than threshold , that circle is white
                binarymap[a] = 1
            else:
                binarymap[a] = 0
    
        #create binary number
        binarymapKeys = list(binarymap.keys())
        binarymapKeys.sort()
        num1 = binarymap[binarymapKeys[0]]
        num2 = binarymap[binarymapKeys[1]]
        num3 = binarymap[binarymapKeys[2]]
        return num1*4 + num2*2 + num3*1

def shape_check(img):
    filter = filter_blue(img)
    gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    
    # list for storing names of shapes
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        # print(cv2.contourArea(contour))
        if cv2.contourArea(contour) < 1000 or cv2.contourArea(contour) > 15000:
            continue 
    
        #print(cv2.contourArea(contour))
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.015 * cv2.arcLength(contour, True), True)
        
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
        cv2.imshow('x',img)
        cv2.waitKey(0)
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
    
        print(len(approx))
        # putting shape name at center of each shape
        if len(approx) == 3:
            return "Triangle"
    
        elif len(approx) == 4:
            return "Rectangle"
    
        elif len(approx) == 5:
            return "Pentagon"
     
        elif len(approx) == 12: #To fix
            return "X"
    
        else:
            return "Circle"

def text_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (2, 2))
    _, threshold = cv2.threshold(gray_blurred, 90, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(threshold,lang='eng', config='--psm 6 outputbase digits')
    return text[0]
# --psm 10 tessedit_char_whitelist=0123456789

def sort_order(first_row, second_row,third_row):
    index = [text_check(e) for e in third_row]
    combined_lists = zip(first_row, second_row,third_row,index)
    sorted_tuples = sorted(combined_lists,key=lambda x:x[3])
    sorted_list1, sorted_list2, sorted_list3,index_sort = zip(*sorted_tuples)
    return sorted_list1, sorted_list2,sorted_list3,index_sort

def segment_image(img):
    height = img.shape[0]
    width = img.shape[1]
    first_row = [img[0:int(height/3), 0:int(width/5) ], img[0:int(height/3), int(width/5):int(width*2/5) ], img[0:int(height/3), int(width*2/5):int(width*3/5) ], img[0:int(height/3), int(width*3/5):int(width*4/5) ], img[0:int(height/3), int(width*4/5):width ]]
    second_row = [img[int(height/3):int(height*2/3), 0:int(width/5) ], img[int(height/3):int(height*2/3), int(width/5):int(width*2/5) ], img[int(height/3):int(height*2/3), int(width*2/5):int(width*3/5) ], img[int(height/3):int(height*2/3), int(width*3/5):int(width*4/5) ], img[int(height/3):int(height*2/3), int(width*4/5):width ]]
    third_row = [img[int(height*2/3):height, 0:int(width/5) ], img[int(height*2/3):height, int(width/5):int(width*2/5) ], img[int(height*2/3):height, int(width*2/5):int(width*3/5) ], img[int(height*2/3):height, int(width*3/5):int(width*4/5) ], img[int(height*2/3):height, int(width*4/5):width ]]
    return first_row, second_row, third_row

def find_largest_rectangle(image):

    # Step 1: Read the image and convert it to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply edge detection (using Canny)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Step 3: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter contours based on area and approximate them as polygons
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter based on the contour area (adjust this threshold as needed)
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Only consider contours with four vertices (approximating rectangles)
                filtered_contours.append(approx)

    # Step 5: Find the largest rectangle among the filtered contours
    largest_rectangle = max(filtered_contours, key=cv2.contourArea)

    # Step 6: Draw the largest rectangle on a copy of the original image
    image_with_rectangle = image.copy()
    cv2.drawContours(image_with_rectangle, [largest_rectangle], 0, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(largest_rectangle)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def filter_blue(img):
    lower_blue = np.array([106, 123, 0])
    upper_blue = np.array([118, 255, 255])
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_filtered_image = cv2.bitwise_and(img, img, mask=blue_mask)
    return blue_filtered_image

def main():
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    img = cv2.imread("images/raw2.png")
    cropped = find_largest_rectangle(img)
    # cv2.imshow('test',cropped)
    first_row, second_row, third_row = segment_image(cropped)
    sorted_list1, sorted_list2,sorted_list3,index_sorted = sort_order(first_row,second_row,third_row)
    for i in range(5):
        print(text_check(third_row[i]),shape_check(second_row[i]))
        #cv2.imshow(f'{i}',filter_blue(second_row[i]))
    cv2.waitKey(0)

if __name__=="__main__":
    main()