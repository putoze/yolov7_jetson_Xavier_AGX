import cv2
import numpy as np
import math

def find_max_Thresh(input_img,flag_list):
    target_img = None

    # Convert to grayscale if gray_flag is true
    if flag_list[0]:
        img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        target_img = img_gray

    # Thresholding if binary_flag is true
    if flag_list[1]:
        _, binary = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY_INV) #95, 255
    #     binary = cv2.adaptiveThreshold(target_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    # cv2.THRESH_BINARY, 11, 2)
        # binary = cv2.adaptiveThreshold(target_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
        #         cv2.THRESH_BINARY, 11, 2)
        target_img = binary

    # Morphological operations if morphology_flag is true
    if flag_list[2]:
        morphologyDisk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphology_img = cv2.morphologyEx(target_img, cv2.MORPH_CLOSE, morphologyDisk)
        target_img = morphology_img

    # Gaussian blur if Gaussblur_flag is true
    if flag_list[3]:
        Gaussblur_img = cv2.GaussianBlur(target_img, (5, 5), 0)
        target_img = Gaussblur_img

    # Sobel edge detection if Sobel_flag is true
    if flag_list[4]:
        sobelX = cv2.Sobel(target_img, cv2.CV_16S, 1, 0, ksize=3)
        sobelY = cv2.Sobel(target_img, cv2.CV_16S, 0, 1, ksize=3)
        sobelX8U = cv2.convertScaleAbs(sobelX)
        sobelY8U = cv2.convertScaleAbs(sobelY)
        Sobel_img = cv2.addWeighted(sobelX8U, 0.5, sobelY8U, 0.5, 0)
        target_img = Sobel_img

    # Canny edge detection if Canny_flag is true
    if flag_list[5]:
        canny = cv2.Canny(target_img, 30, 150)
        target_img = canny

    # Find contours if Contours_flag is true
    if flag_list[6]:
        contours, _ = cv2.findContours(target_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(target_img, contours, 3, (0, 255, 0), 3)


    # to show image or not
    # if flag_list[1]:
    #     cv2.imshow("binary Image", binary)
    # if flag_list[2]:
    #     cv2.imshow("morphology Image", morphology_img)
    # if flag_list[3]:
    #     cv2.imshow("Gaussian Image", Gaussblur_img)
    # if flag_list[4]:
    #     cv2.imshow("Sobel_img Image", Sobel_img)
    # if flag_list[5]:
    #     cv2.imshow("canny Image", canny)
    # if flag_list[6]:
    #     cv2.imshow("contours Image", target_img)

    # find max area
    maxArea = 0
    max_countuor = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea:
            maxArea = area
            max_countuor = contour

    # Find best areas
    # print("max area", maxArea)

    if maxArea != 0:
        # momentsPupilThresh = cv2.moments(max_countuor)
        # center = (int(momentsPupilThresh["m10"] / momentsPupilThresh["m00"]),
        #              int(momentsPupilThresh["m01"] / momentsPupilThresh["m00"]))

        # cv2.circle(img, center, 3, (0, 0, 255), -1)
        contour_pt_array = np.array(max_countuor, dtype=np.int32)

        # Avoid to break the system
        if(contour_pt_array.shape[0] < 5):
            return 0
        
        elPupilThresh = cv2.fitEllipse(contour_pt_array)

        # print("elPupilThresh")
        # print("Center:",elPupilThresh[0])
        # print("Size:" ,elPupilThresh[1])
        # print("Angle:" ,elPupilThresh[2])

        # Color = (0, 255, 0)  # Green color
        # thickness = 2
        # center = (int(elPupilThresh[0][0]),int(elPupilThresh[0][1]))
        # cv2.ellipse(draw_img, elPupilThresh, Color, thickness)
        # cv2.circle(draw_img, center, 3, (0, 0, 255), -1)
        
        # final_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        return elPupilThresh
    else :
        return None

    
        
        
        

