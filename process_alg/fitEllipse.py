import cv2
import numpy as np
import math

def find_max_Thresh(input_img,flag_list):
    target_img = None

    # Convert to grayscale if gray_flag is true
    if flag_list[0]:
        img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        target_img = img_gray
    
    # Normalize image
    # min_pixel_value = np.min(img_gray)
    # max_pixel_value = np.max(img_gray)
    # normalized_image = ((img_gray - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255).astype(np.uint8)
    # target_img = normalized_image

    equalized_image = cv2.equalizeHist(img_gray)
    target_img = equalized_image

    # Thresholding if binary_flag is true
    if flag_list[1]:
        binary = cv2.adaptiveThreshold(target_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY, 11, 2)
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
    #     cv2.imshow("binary Image", cv2.resize(binary,(100,100)))
    # if flag_list[2]:
    #     cv2.imshow("morphology Image", cv2.resize(morphology_img,(100,100)))
    # if flag_list[3]:
    #     cv2.imshow("Gaussian Image", cv2.resize(Gaussblur_img,(100,100)))
    # if flag_list[4]:
    #     cv2.imshow("Sobel_img Image", cv2.resize(Sobel_img,(100,100)))
    # if flag_list[5]:
    #     cv2.imshow("canny Image", cv2.resize(canny,(100,100)))
    # if flag_list[6]:
    #     cv2.imshow("contours Image", cv2.resize(target_img,(100,100)))
    # cv2.imshow("equalized Image", cv2.resize(equalized_image,(100,100)))

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

    
def draw_Ellipse_fit(img,pupil,elPupilThresh):
    #(Gray,Binary,Morphological,Gaussian blur,Sobel,Canny,Find contours)

    # update elPupilThresh into golbal image
    center = (int(elPupilThresh[0][0] + pupil[0]), int(elPupilThresh[0][1] + pupil[1]))
    new_elPupilThresh_left = (center,elPupilThresh[1],elPupilThresh[2])
    # cv2.ellipse(img, new_elPupilThresh_left, (0, 255, 0), 2)
    cv2.circle(img, center, 3, (0, 0, 255), -1)
    # resize image into top left corner
    return center

def find_ellipse_point(num_points,elthresh):
    center,axes,angle = elthresh
    angle_step = 360 / num_points
    ellipse_points = []
    for i in range(num_points):
        angle_deg = angle + angle_step * i
        angle_rad = np.radians(angle_deg)
        x_point = center[0] + (axes[0] / 2) * np.cos(angle_rad)
        y_point = center[1] + (axes[1] / 2) * np.sin(angle_rad)
        ellipse_points.append([x_point, y_point])

    return np.array(ellipse_points)

def draw_ellipse_point(img,ellipse_points,pupil_roi):
    img_out = img
    for pt in ellipse_points:
        point = (int(pupil_roi[0] + pt[0]),int(pupil_roi[1] + pt[1]))
        cv2.circle(img_out, point, 2, (0, 255, 0), -1)

    return img_out
        
        

