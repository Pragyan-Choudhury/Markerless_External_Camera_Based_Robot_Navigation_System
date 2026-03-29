import cv2

url = "http://192.168.0.104:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print(" Cannot open stream")
else:
    print(" Stream opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Stream", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()