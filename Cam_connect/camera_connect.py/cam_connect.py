import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v') 
writer = cv2.VideoWriter("recoding.mp4", fourcc, 30.0, (1280,720))
is_recording = False

while True:
    return_value, frame = cap.read()
    if return_value:
        cv2.imshow("video", frame)
        if is_recording:
            writer.write(frame) #writing frame into the file
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break #quit
    elif key == ord('r'):
        is_recording = not is_recording
        print(f"Recording {is_recording}")
        
cap.release()
writer.release()
cv2.destroyAllWindows()
