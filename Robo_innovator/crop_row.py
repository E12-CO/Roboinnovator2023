import cv2
def crop_row(cropped_image): 
    for i in range(5):
        start_col = int(cropped_image.shape[1] * (i / 5))
        end_col = int(cropped_image.shape[1] * ((i + 1) / 5))
        for j in range(3):
            start_row = int(cropped_image.shape[0] * (j / 3))
            end_row = int(cropped_image.shape[0] * ((j + 1) / 3))
            segment_image = cropped_image[start_row:end_row, start_col:end_col]
            cv2.imwrite(f"images/{i + 1}_{j+1}.png", segment_image)