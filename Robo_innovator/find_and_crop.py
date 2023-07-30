import cv2
import math
def find_largest_rectangle(image_path):
    # Step 1: Read the image and convert it to grayscale
    image = cv2.imread(image_path)
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
    print(largest_rectangle)

    # Step 6: Draw the largest rectangle on a copy of the original image
    image_with_rectangle = image.copy()
    cv2.drawContours(image_with_rectangle, [largest_rectangle], 0, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(largest_rectangle)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def crop_row_col(cropped_image): 
    for i in range(5):
        start_col = int(cropped_image.shape[1] * (i / 5))
        end_col = int(cropped_image.shape[1] * ((i + 1) / 5))
        for j in range(3):
            start_row = int(cropped_image.shape[0] * (j / 3))
            end_row = int(cropped_image.shape[0] * ((j + 1) / 3))
            segment_image = cropped_image[start_row:end_row, start_col:end_col]
            cv2.imwrite(f"images/{i + 1}_{j+1}.png", segment_image)

if __name__ == "__main__":
    image_path = "images/raw2.png"  # Replace with the path to your image
    cropped_image = find_largest_rectangle(image_path)
    crop_row_col(cropped_image)

