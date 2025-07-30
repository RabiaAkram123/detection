import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_intrest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    img = np.copy(img)
    blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
    return img

def process(image):
    width = image.shape[1]
    height = image.shape[0]

    region_of_intrest1 = [
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.8 * height)),
        (int(0.6 * width), int(0.8 * height)),
        (int(0.9 * width), height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_img = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_intrest(canny_img, vertices=np.array([region_of_intrest1], np.int32))
    lines = cv2.HoughLinesP(cropped_image, 6, np.pi / 60, threshold=160, minLineLength=40, maxLineGap=25)
    if lines is not None:
        image = draw_lines(image, lines)
    return image

cap = cv2.VideoCapture("road.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Road Lines", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
