import numpy as np
import cv2
import pytesseract
import random
def binarycheck(img,found_numbers):
    # Convert to grayscale.
    binarymapKeys = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (2, 2))
    # Blur using 3 * 3 kernel.
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

            threshold = 0.75 #threshold to determine if circle is white/black
            dumcnt = 0

            for h in range(dumimg.shape[0]):
                for w in range(dumimg.shape[1]):
                    if int(dumimg[h,w][0]) + int(dumimg[h,w][1])+ int(dumimg[h,w][2]) > 500:  #if pixel is white
                        dumcnt+=1
            res = float((dumcnt)/(dumimg.shape[0]*dumimg.shape[1])) 
            # print(res)
            if res > threshold:
                #if result is higher than threshold , that circle is white
                binarymap[a] = 1
            else:
                binarymap[a] = 0
    
        #create binary number
        binarymapKeys = list(binarymap.keys())
        binarymapKeys.sort()
    if len(binarymapKeys) == 0:
        num1 = random.randint(0,1)
        num2 = (num1 + 1) % 2
        num3 = random.randint(0,1)
        x = num1*4 + num2*2 + num3*1
        while x in found_numbers or (x == 0 or x == 6):
            x = random.randint(1,5)
        found_numbers.append(x) 
        return x
    if len(binarymapKeys) == 1 :
        num1 = binarymap[binarymapKeys[0]]
        num2 = (num1 + 1) % 2 
        num3 = random.randint(0,1) 
        x = num1*4 + num2*2 + num3*1
        while x in found_numbers or (x == 0 or x == 6):
            x = random.randint(1,5)
        found_numbers.append(x) 
        return x
    if len(binarymapKeys) == 2 :
        num1 = binarymap[binarymapKeys[0]]
        num2 = binarymap[binarymapKeys[1]]
        num3 = 0
        x = num1*4 + num2*2 + num3*1
        while x in found_numbers or (x == 0 or x == 6):
            x = random.randint(1,5)
        found_numbers.append(x) 
        return x
    num1 = binarymap[binarymapKeys[0]]
    num2 = binarymap[binarymapKeys[1]]
    num3 = binarymap[binarymapKeys[2]]
    x = num1*4 + num2*2 + num3*1
    while x in found_numbers or (x == 0 or x == 6):
        x = random.randint(1,5)
    found_numbers.append(x) 
    return x
    
def text_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erd = cv2.erode(gray,None,iterations=1)
    gray_blurred = cv2.blur(erd, (5, 5))
    _, threshold = cv2.threshold(gray_blurred, 130, 255, cv2.THRESH_BINARY)
    # cv2.imshow('t',threshold)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(threshold,lang='eng', config='--psm 6 outputbase digits')
    if len(text) == 0 :
        return 'None'
    if text[0] in 'i]':
        return '1'
    if text[0] == 'A' :
        return '4'
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
    edges = cv2.Canny(gray_image, 220, 255, apertureSize=3)

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
    if len(filtered_contours) == 0 :
        return image
    largest_rectangle = max(filtered_contours, key=cv2.contourArea)

    # Step 6: Draw the largest rectangle on a copy of the original image
    image_with_rectangle = image.copy()
    cv2.drawContours(image, [largest_rectangle], 0, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(largest_rectangle)
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    return image[y:y+h, x:x+w]

def filter_blue(img):
    lower_blue = np.array([30, 40, 80])
    upper_blue = np.array([179, 255, 204])
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_filtered_image = cv2.bitwise_and(img, img, mask=blue_mask)
    # cv2.imshow('t',blue_filtered_image)
    # cv2.waitKey(0)
    return blue_filtered_image

def Area(img):
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
        if cv2.contourArea(contour) < 500 or cv2.contourArea(contour) > 2000:
            continue 
    
        #print(cv2.contourArea(contour))
        # cv2.approxPloyDP() function to approximate the shapes
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
        # cv2.imshow('s',img)
        # cv2.waitKey(0)
        area = cv2.contourArea(contour)
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        return int(area)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
    # cv2.imshow('s',img)
    return int(random.randint(0,5000))

def classify_shape(image_list):
    image_raw = image_list.copy()
    image_list.sort()
    result = []
    for i in range(5):
        if image_list.index(image_raw[i]) == 0 :
            result.append("X")
        elif image_list.index(image_raw[i]) == 1:
            result.append("Triangle")
        elif image_list.index(image_raw[i]) == 3:
            result.append("Pentagon")
        elif image_list.index(image_raw[i]) == 2:
            result.append("Circle")
        elif image_list.index(image_raw[i]) == 4:
            result.append("Rectangle")
    return (result)

def generate_output(frame):
    store_shape = []
    store_numbers = []
    store_binary = []
    image = find_largest_rectangle(frame)
    found_numbers = []
    message = ""
    scale_factor = 2
    new_width = image.shape[1] * scale_factor
    new_height = image.shape[0] * scale_factor
    new_dimensions = (new_width, new_height)
    enlarged_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    first_row, second_row, third_row = segment_image(enlarged_image)
    for i in range(5):
        store_shape.append(Area(second_row[i]))
        store_numbers.append(text_check(third_row[i]))
        store_binary.append(binarycheck(first_row[i],found_numbers))
    result = classify_shape(store_shape)

    for ind,i in enumerate(store_numbers):
        if i not in "12345" : 
            x = random.randint(1,5)
            while x in store_numbers:
                x = random.randint(1,5)
            store_numbers[ind] = x        
    for i in range(5):
        message += str(store_binary[i]) + ' ' + str(result[i]) + ' ' + str(store_numbers[i])
        if i != 4 :
            message += ','
    return message

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = find_largest_rectangle(frame)
        cv2.imshow('frame',image)
        if cv2.waitKey(10) == ord('s'):
            break
        if cv2.waitKey(10) == ord('q'):
            message = generate_output(frame)
            print(message)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    main()