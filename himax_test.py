import numpy as np
import cv2

#================================================================================== image ===


#================================================================================== video ===
cap = cv2.VideoCapture(9)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # write the flipped frame
        cv2.imshow('frame',frame)
        
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()