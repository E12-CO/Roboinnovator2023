import cv2
import numpy as np
import random
import pytesseract

def Area(img,images,theshold_area,minArea_area,maxAera_area):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images[6] = gray
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, theshold_area, 255, cv2.THRESH_BINARY)
    images[7] = threshold    
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    # list for storing names of shapes
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if cv2.contourArea(contour) < minArea_area or cv2.contourArea(contour) > maxAera_area:
            continue 
    
        #print(cv2.contourArea(contour))
        # cv2.approxPloyDP() function to approximate the shapes
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
        images[8] = img
        # cv2.imshow('s',img)
        # cv2.waitKey(0)
        area = cv2.contourArea(contour)
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        return int(area)
    images[8] = img
    return
    # cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
    # # cv2.imshow('s',img)
    # images[8] = img
    # return int(random.randint(0,5000))

def text_check(img,images,erodeiter_text,kernelsize_text,threshold_text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erd = cv2.erode(gray,None,iterations=erodeiter_text)
    images[9] = erd
    gray_blurred = cv2.blur(erd, (kernelsize_text, kernelsize_text))
    images[10] = gray_blurred
    _, threshold = cv2.threshold(gray_blurred, threshold_text, 255, cv2.THRESH_BINARY)
    images[11] = threshold
    text = pytesseract.image_to_string(threshold,lang='eng', config='--psm 6 outputbase digits')
    if len(text) == 0 :
        return 'None'
    if text[0] in 'i]':
        return '1'
    if text[0] == 'A' :
        return '4'
    return text[0]

def segment_image(img):
    height = img.shape[0]
    width = img.shape[1]
    first_row = [img[0:int(height/3), 0:int(width/5) ], img[0:int(height/3), int(width/5):int(width*2/5) ], img[0:int(height/3), int(width*2/5):int(width*3/5) ], img[0:int(height/3), int(width*3/5):int(width*4/5) ], img[0:int(height/3), int(width*4/5):width ]]
    second_row = [img[int(height/3):int(height*2/3), 0:int(width/5) ], img[int(height/3):int(height*2/3), int(width/5):int(width*2/5) ], img[int(height/3):int(height*2/3), int(width*2/5):int(width*3/5) ], img[int(height/3):int(height*2/3), int(width*3/5):int(width*4/5) ], img[int(height/3):int(height*2/3), int(width*4/5):width ]]
    third_row = [img[int(height*2/3):height, 0:int(width/5) ], img[int(height*2/3):height, int(width/5):int(width*2/5) ], img[int(height*2/3):height, int(width*2/5):int(width*3/5) ], img[int(height*2/3):height, int(width*3/5):int(width*4/5) ], img[int(height*2/3):height, int(width*4/5):width ]]
    return first_row, second_row, third_row

def binarycheck(img,images,kernelsize_bin,minraduis_bin,maxraduis_bin):
    # Convert to grayscale.
    binarymapKeys = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images[3] = gray
    gray_blurred = cv2.blur(gray, (kernelsize_bin, kernelsize_bin))
    images[4] = gray_blurred
    # Blur using 3 * 3 kernel.
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = minraduis_bin, maxRadius = maxraduis_bin)
    
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        binarymap = dict()

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img,(a,b),r,(255,0,0),2)
            halfr = int(r/2)
            dumimg = img[b-halfr:b+halfr, a-halfr:a+halfr]

            threshold = 0.75 #threshold to determine if circle is white/black
            dumcnt = 0

            for h in range(dumimg.shape[0]):
                for w in range(dumimg.shape[1]):
                    if int(dumimg[h,w][0]) + int(dumimg[h,w][1])+ int(dumimg[h,w][2]) > 500:  #if pixel is white
                        dumcnt+=1
            if dumimg.shape[0]*dumimg.shape[1] == 0 :
                return
            res = float((dumcnt)/(dumimg.shape[0]*dumimg.shape[1])) 
            # print(res)
            if res > threshold:
                #if result is higher than threshold , that circle is white
                binarymap[a] = 1
            else:
                binarymap[a] = 0
        images[5] = img
        #create binary number
        binarymapKeys = list(binarymap.keys())
        binarymapKeys.sort()
    images[5] = img

def find_largest_rectangle(image,images,area_largest,epsilon_largest,canny_largest,canny2_largest):

    # Step 1: Read the image and convert it to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images[0] = gray_image
    # Step 2: Apply edge detection (using Canny)
    edges = cv2.Canny(gray_image, canny2_largest,canny_largest, apertureSize=3)
    images[1] = edges
    # Step 3: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter contours based on area and approximate them as polygons
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_largest:  # Filter based on the contour area (adjust this threshold as needed)
            epsilon = epsilon_largest/100 * cv2.arcLength(contour, True)
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
    images[2] = image
    return image[y:y+h, x:x+w]

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

def show_tunning(frame,images,indeximage,area_largest,epsilon_largest,canny_largest,canny2_largest,kernelsize_bin,minraduis_bin,maxraduis_bin,theshold_area,minArea_area,maxAera_area,erodeiter_text,kernelsize_text,threshold_text):
    croped = find_largest_rectangle(frame,images,area_largest,epsilon_largest,canny_largest,canny2_largest)
    first_row, second_row, third_row = segment_image(croped)
    binarycheck(first_row[indeximage],images,kernelsize_bin,minraduis_bin,maxraduis_bin)
    Area(second_row[indeximage],images,theshold_area,minArea_area,maxAera_area)
    text_check(third_row[indeximage],images,erodeiter_text,kernelsize_text,threshold_text)
    num_rows = 4
    num_cols = 3
    image_height = 200
    image_width = 200
    canvas_height = image_height * num_rows
    canvas_width = image_width * num_cols
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            if idx < len(images):
                resized_image = cv2.resize(images[idx], (image_width, image_height))
                if len(resized_image.shape) == 2 :
                    resized_image = cv2.resize(resized_image, (resized_image.shape[1], resized_image.shape[0])).astype(np.uint8)
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
                canvas[r * image_height : (r + 1) * image_height, c * image_width : (c + 1) * image_width,:] = resized_image
    return canvas
def nothing(x):
    pass
def main():
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img',600,600)
    # index images
    cv2.createTrackbar('indeximage', 'img', 0, 4, nothing)

    # largest rectangle
    cv2.createTrackbar('area_largest', 'img', 0, 20000, nothing)
    cv2.createTrackbar('epsilon_largest', 'img', 0, 100, nothing)
    cv2.createTrackbar('canny_largest', 'img', 0, 255,nothing)
    cv2.createTrackbar('canny2_largest', 'img', 0, 255,nothing)

    # tuning binary
    cv2.createTrackbar('kernelsize_bin', 'img', 0, 20, nothing)
    cv2.createTrackbar('minraduis_bin', 'img', 0, 250, nothing)
    cv2.createTrackbar('maxraduis_bin', 'img', 0, 250, nothing)

    # Area
    cv2.createTrackbar('theshold_area', 'img', 0, 255, nothing)
    cv2.createTrackbar('minArea_area', 'img', 0, 20000, nothing)
    cv2.createTrackbar('maxAera_area', 'img', 0, 20000,nothing)

    # iterations teseract
    cv2.createTrackbar('erodeiter_text', 'img', 0, 10, nothing)
    cv2.createTrackbar('kernelsize_text', 'img', 0, 10, nothing)
    cv2.createTrackbar('threshold_text', 'img', 0, 255,nothing)

    # set default values
    cv2.setTrackbarPos('indeximage', 'img', 0)
    cv2.setTrackbarPos('area_largest', 'img', 1000)
    cv2.setTrackbarPos('epsilon_largest', 'img',10)
    cv2.setTrackbarPos('canny_largest', 'img', 150)
    cv2.setTrackbarPos('canny2_largest', 'img', 50)
    cv2.setTrackbarPos('kernelsize_bin', 'img', 2)
    cv2.setTrackbarPos('minraduis_bin', 'img', 1)
    cv2.setTrackbarPos('maxraduis_bin', 'img', 40)
    cv2.setTrackbarPos('theshold_area', 'img', 127)
    cv2.setTrackbarPos('minArea_area', 'img', 500)
    cv2.setTrackbarPos('maxAera_area', 'img', 2000)
    cv2.setTrackbarPos('erodeiter_text', 'img', 1)
    cv2.setTrackbarPos('kernelsize_text', 'img', 5)
    cv2.setTrackbarPos('threshold_text', 'img', 130)

    indeximage = area_largest = epsilon_largest = canny_largest = canny2_largest = 0
    kernelsize_bin = minraduis_bin = maxraduis_bin = theshold_area = 0
    minArea_area = maxAera_area = erodeiter_text = kernelsize_text = threshold_text = 0
    while (1) : 
        indeximage = cv2.getTrackbarPos('indeximage','img')
        area_largest = cv2.getTrackbarPos('area_largest', 'img')
        epsilon_largest = cv2.getTrackbarPos('epsilon_largest', 'img')
        canny_largest = cv2.getTrackbarPos('canny_largest', 'img')
        canny2_largest = cv2.getTrackbarPos('canny2_largest', 'img')
        kernelsize_bin = cv2.getTrackbarPos('kernelsize_bin', 'img')
        minraduis_bin = cv2.getTrackbarPos('minraduis_bin', 'img')
        maxraduis_bin = cv2.getTrackbarPos('maxraduis_bin', 'img')
        theshold_area = cv2.getTrackbarPos('theshold_area', 'img')
        minArea_area = cv2.getTrackbarPos('minArea_area', 'img')
        maxAera_area = cv2.getTrackbarPos('maxAera_area', 'img')
        erodeiter_text = cv2.getTrackbarPos('erodeiter_text', 'img')
        kernelsize_text = cv2.getTrackbarPos('kernelsize_text', 'img')
        threshold_text = cv2.getTrackbarPos('threshold_text', 'img')
        image_path = 'images/raw2.png'
        image = cv2.imread(image_path)
        images = [0]*12
        canvas = show_tunning(image,images,indeximage,area_largest,epsilon_largest,canny_largest,canny2_largest,kernelsize_bin,minraduis_bin,maxraduis_bin,theshold_area,minArea_area,maxAera_area,erodeiter_text,kernelsize_text,threshold_text)
        cv2.imshow('window',canvas)
        # cv2.imshow('img',)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
# def test():
#     images = [0]*12
#     image = cv2.imread('images/raw2.png')
#     img = find_largest_rectangle(image,images,1000,0.1,50)
#     cv2.imshow('window',img)
#     cv2.waitKey(0)

if __name__ == '__main__':
    main()