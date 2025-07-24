import cv2

# Create background subtractor (AI technique)
backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)
object_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    # Display object count
    cv2.putText(frame, f'Objects Detected: {count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Object Counter AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
