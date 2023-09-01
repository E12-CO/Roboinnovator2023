import cv2
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    sample_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"images/sample{sample_number}.png",frame)
            sample_number += 1

if __name__ == '__main__':
    main()