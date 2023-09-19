import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import re
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from numpy.lib.function_base import _quantile_unchecked
from matplotlib import pyplot as plt
from model_6DRepNet import SixDRepNet
import utils_with_6D
import matplotlib
matplotlib.use('TkAgg')

import torch.nn as nn
from utils_ten.display import open_window, show_fps, set_display
from torch.autograd import Variable
from torchvision import transforms
import torchvision

from PIL import Image
from utils_L2CS import draw_gaze
from PIL import Image, ImageOps
from imutils import face_utils
import dlib

from model_L2CS import L2CS
from process_alg.fitEllipse import *
import process_alg.gaze_modelbased as GM 
import process_alg.gaze as gaze_util

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Self add
WINDOW_NAME = 'Yolov7-Tiny Demo'

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    show_text = True
    open_window(WINDOW_NAME,'yolov7-tiny')

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1


    transformations_6D = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 6D_Repnet
    snapshot_path_6D = '../weights/6DRepNet/6DRepNet_300W_LP_AFLW2000.pth'
    model_6DRepNet = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    saved_state_dict = torch.load(os.path.join(
        snapshot_path_6D), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model_6DRepNet.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model_6DRepNet.load_state_dict(saved_state_dict)
    model_6DRepNet.to(device)

    # Test the Model
    model_6DRepNet.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    # End 6D_Repnet

    # landmark
    model_landmark = '../weights/landmark-model/shape_predictor_5_face_landmarks.dat'
    predictor = dlib.shape_predictor(model_landmark)

    # Self-define global parameter
    # ---------------------------
    # ------ face imformation ------
    nose_center_point = (0,0)
    left_eye_center = (0,0)
    right_eye_center = (0,0)
    eye_w_roi = 100
    eye_h_roi = 50
    #------ Thresh ------
    El_right_eye_thresh_global = ()
    El_left_eye_thresh_global = ()
    El_right_iris_thresh_global = ()
    El_left_iris_thresh_global = ()
    num_iris_landmark = 8
    #------ eye img ------
    right_eye_img = cv2.imread("./test_image/eye/3.png")  
    right_eye_img = cv2.resize(right_eye_img,(eye_w_roi,eye_h_roi))
    left_eye_img = cv2.imread("./test_image/eye/2.png")  
    left_eye_img = cv2.resize(left_eye_img,(eye_w_roi,eye_h_roi))
    #------ length ------
    right_eye_length = 0
    left_eye_length = 0
    #------ put txt ------
    base_txt_height = 35 + eye_h_roi
    gap_txt_height = 35
    #------ gaze flag ------
    gaze_model_flag = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
    

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            start = time.time()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ------- Main Algorithm ------
                face_max = 0
                eye_right = []
                eye_left  = []
                pupil_left = []
                pupil_right = []
                mouse_roi = []

                driver_face_local = []
                # find driver face
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == 'face':
                        bb = [int(x) for x in xyxy]
                        face_area = (bb[2] - bb[0])*(bb[3]-bb[1])
                        if(face_area > face_max) :
                            face_max = face_area
                            driver_face_local = bb

                if len(driver_face_local) == 0:
                    break

                # find all boundary of face     
                for *xyxy, conf, cls in reversed(det):
                    bb = [int(x) for x in xyxy]
                    center = (bb[0]+(bb[2]-bb[0])/2,bb[1]+(bb[3]-bb[1])/2)
                    # if cs of landmark center inside the driver face
                    if (center[0] > driver_face_local[0] and center[0] < driver_face_local[2]) \
                    and (center[1] > driver_face_local[1] and center[1] < driver_face_local[3]):
                        if names[int(cls)] == 'eye' and bb[3] < nose_center_point[1]:
                            if (center[0] < nose_center_point[0]):
                                eye_left = bb
                            else:
                                eye_right = bb
                        if names[int(cls)] == 'pupil' and bb[3] < nose_center_point[1]:
                            if (center[0] < nose_center_point[0]):
                                pupil_left = bb
                            else :
                                pupil_right = bb
                        if names[int(cls)] == 'nose' :
                            nose_center_point = center
                        if names[int(cls)] == 'mouse':
                            mouse_roi = bb

                # 6DRepNet
                x_min,y_min,x_max,y_max = driver_face_local
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = im0[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations_6D(img)

                img = torch.Tensor(img[None, :]).to(device)

                R_pred = model_6DRepNet(img)

                euler = utils_with_6D.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # utils_with_6D.plot_pose_cube(im0,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                #     x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
                height, width = im0.shape[:2]
                tdx = width - 70
                tdy = 70

                # R_headpose = utils_with_6D.get_R(r_pred_deg,y_pred_deg,p_pred_deg)
                utils_with_6D.draw_axis(im0,y_pred_deg,p_pred_deg,r_pred_deg,tdx,tdy, size = 50)
                utils_with_6D.draw_gaze_6D(nose_center_point,im0,y_pred_deg,p_pred_deg,color=(0,0,255))

                # End 6DRepNet

                # pupil left
                flag_list = [1,1,1,1,1,1,1]

                # ellipse fit eye and define eye region
                if len(eye_left) > 0:
                    left_eye_img = im0[eye_left[1]:eye_left[3],eye_left[0]:eye_left[2],:]
                    # El_left_eye_thresh = find_max_Thresh(left_eye_img,flag_list)
                    # El_left_eye_thresh_global = ((El_left_eye_thresh[0][0] + eye_left[0], El_left_eye_thresh[0][1] \
                    #                               + eye_left[1]),El_left_eye_thresh[1],El_left_eye_thresh[2])
                    # cv2.ellipse(im0, El_left_eye_thresh_global, (255, 0, 0), 2)
                    # left_eye_center = El_left_eye_thresh_global[0]
                    # left_eye_length = El_left_eye_thresh_global[1][0]
                    left_eye_center = (eye_left[0]+(eye_left[2]-eye_left[0])/2,eye_left[1]+(eye_left[3]-eye_left[1])/2)
                    left_eye_length = (eye_left[2]-eye_left[0])/2
                if len(eye_right) > 0:
                    right_eye_img = im0[eye_right[1]:eye_right[3],eye_right[0]:eye_right[2],:]
                    # El_right_eye_thresh = find_max_Thresh(right_eye_img,flag_list)
                    # El_right_eye_thresh_global = ((El_right_eye_thresh[0][0] + eye_right[0], El_right_eye_thresh[0][1] \
                    #                               + eye_right[1]),El_right_eye_thresh[1],El_right_eye_thresh[2])
                    # cv2.ellipse(im0, El_right_eye_thresh_global, (255, 0, 0), 2)
                    # right_eye_center = El_right_eye_thresh_global[0]
                    # right_eye_length = El_right_eye_thresh_global[1][0]
                    right_eye_center = (eye_right[0]+(eye_right[2]-eye_right[0])/2,eye_right[1]+(eye_right[3]-eye_right[1])/2)
                    right_eye_length = (eye_right[2]-eye_right[0])/2

                # ellipse fit left pupil and gaze estimate
                if len(pupil_left) > 0 and len(eye_left) > 0:
                    # pupil_left_img = im0[pupil_left[1]:pupil_left[3],pupil_left[0]:pupil_left[2],:]
                    # pupil_left_center = (int((pupil_left[0]+pupil_left[2])/2),int((pupil_left[1]+pupil_left[3])/2))
                    # radius_left = int(min(pupil_left[3]-pupil_left[1],pupil_left[2]-pupil_left[0])/4*3)
                    # # create a circle to reduce shade effect
                    # cv2.circle(pupil_left_img,pupil_left_center,radius_left,(0,0,0),thickness=cv2.FILLED)
                    # El_left_iris_thresh = find_max_Thresh(pupil_left_img,flag_list)
                    
                    # if El_left_iris_thresh != None:
                    #     El_left_iris_thresh_global = ((El_left_iris_thresh[0][0] + pupil_left[0], El_left_iris_thresh[0][1] \
                    #                                 + pupil_left[1]),El_left_iris_thresh[1],El_left_iris_thresh[2])
                    #     left_iris_ldmks = find_ellipse_point(num_iris_landmark,El_left_iris_thresh_global)
                    #     im0 = draw_ellipse_point(im0,left_iris_ldmks,El_left_iris_thresh_global)
                    #     left_gaze = GM.estimate_gaze_from_landmarks(left_iris_ldmks, El_left_iris_thresh_global[0], left_eye_center, left_eye_length)
                    #     left_gaze = left_gaze.reshape(1, 2)

                    #     left_gaze[0][1] = -left_gaze[0][1]
                    #     im0 = gaze_util.draw_gaze(im0,El_left_iris_thresh_global[0],left_gaze[0])
                    if gaze_model_flag == 1 and abs(y_pred_deg[0].item()) < 30.0:
                        # facelandmark
                        rect = dlib.rectangle(driver_face_local[0],driver_face_local[1],driver_face_local[2],driver_face_local[3])
                        shape = predictor(im0, rect)
                        shape = face_utils.shape_to_np(shape)
                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # left
                        if shape[0][0] > eye_left[0]:
                            cv2.circle(im0, (shape[0][0], shape[0][1]), 1, (0, 0, 255), -1)
                        
                        pupil_left_center = (int((pupil_left[0]+pupil_left[2])/2),int((pupil_left[1]+pupil_left[3])/2))
                        radius_left = (int(pupil_left[2]-pupil_left[0]),int(pupil_left[3]-pupil_left[1]))
                        left_iris_ldmks = find_yolo_ellipse_point(num_iris_landmark,pupil_left_center,radius_left)
                        im0 = draw_yolo_ellipse_point(im0,left_iris_ldmks,pupil_left_center)
                        left_gaze = GM.estimate_gaze_from_landmarks(left_iris_ldmks, pupil_left_center, left_eye_center, left_eye_length)
                    
                        left_gaze = left_gaze.reshape(1, 2)
                        left_gaze[0][1] = -left_gaze[0][1]
                        im0 = gaze_util.draw_gaze(im0,pupil_left_center,left_gaze[0])


                # ellipse fit right pupil and gaze estimate
                if len(pupil_right) > 0 and len(eye_right) > 0:
                    # pupil_right_img = im0[pupil_right[1]:pupil_right[3],pupil_right[0]:pupil_right[2],:]
                    # pupil_right_center = (int((pupil_right[0]+pupil_right[2])/2),int((pupil_right[1]+pupil_right[3])/2))
                    # radius_right = int(min(pupil_right[3]-pupil_right[1],pupil_right[2]-pupil_right[0])/4*3)
                    # # create a circle to reduce shade effect
                    # cv2.circle(pupil_right_img,pupil_right_center,radius_right,(0,0,0),thickness=cv2.FILLED)
                    # El_right_iris_thresh = find_max_Thresh(pupil_right_img,flag_list)
                    
                    # if El_right_iris_thresh != None:
                    #     El_right_iris_thresh_global = ((El_right_iris_thresh[0][0] + pupil_right[0], El_right_iris_thresh[0][1] \
                    #                                 + pupil_right[1]),El_right_iris_thresh[1],El_right_iris_thresh[2])
                    #     right_iris_ldmks = find_ellipse_point(num_iris_landmark,El_right_iris_thresh_global)
                    #     im0 = draw_ellipse_point(im0,right_iris_ldmks,El_right_iris_thresh_global)
                    #     right_gaze = GM.estimate_gaze_from_landmarks(right_iris_ldmks, El_right_iris_thresh_global[0], right_eye_center, right_eye_length)
                    #     right_gaze = right_gaze.reshape(1, 2)

                    #     right_gaze[0][1] = -right_gaze[0][1]
                    #     im0 = gaze_util.draw_gaze(im0,El_right_iris_thresh_global[0],right_gaze[0])
                    if gaze_model_flag == 1:
                        pupil_right_center = (int((pupil_right[0]+pupil_right[2])/2),int((pupil_right[1]+pupil_right[3])/2))
                        radius_right = (int(pupil_right[2]-pupil_right[0]),int(pupil_right[3]-pupil_right[1]))
                        right_iris_ldmks = find_yolo_ellipse_point(num_iris_landmark,pupil_right_center,radius_right)
                        im0 = draw_yolo_ellipse_point(im0,right_iris_ldmks,pupil_right_center)
                        right_gaze = GM.estimate_gaze_from_landmarks(right_iris_ldmks, pupil_right_center, right_eye_center, right_eye_length)

                        right_gaze = right_gaze.reshape(1, 2)
                        right_gaze[0][1] = -right_gaze[0][1]
                        im0 = gaze_util.draw_gaze(im0,pupil_right_center,right_gaze[0])


                # # headpose + gaze
                # if eye_gaze != []:
                #     eye_gaze = np.mean(eye_gaze, axis=0)
                #     y_pred_deg += 
                #     R_headpose = utils_with_6D.get_R(r_pred_deg,y_pred_deg,p_pred_deg)

                # update eye image
                if len(eye_left) > 0:
                    # print("eye_left",eye_left)
                    left_eye_img = im0[eye_left[1]:eye_left[3],eye_left[0]:eye_left[2],:]
                    left_eye_img = cv2.resize(left_eye_img,(eye_w_roi,eye_h_roi))
                if len(eye_right) > 0:
                    # print("eye_right",eye_right)
                    right_eye_img = im0[eye_right[1]:eye_right[3],eye_right[0]:eye_right[2],:]
                    right_eye_img = cv2.resize(right_eye_img,(eye_w_roi,eye_h_roi))

                # put eye image on left top
                im0[0:eye_h_roi,0:eye_w_roi,:] = left_eye_img
                im0[0:eye_h_roi,eye_w_roi:2*eye_w_roi,:]  = right_eye_img
                # im0[eye_h_roi:2*eye_h_roi,0:eye_w_roi,:] = rotation_left_eye_img
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # print(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            end = time.time()
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if show_text:
                pitch_str = str(round(p_pred_deg[0].item(), 3))
                yaw_str = str(-(round(y_pred_deg[0].item(), 3)))
                roll_str = str(round(r_pred_deg[0].item(), 3))
                #(img, text, org, fontFace, fontScale, color, thickness, lineType)
                next_txt_height = base_txt_height
                cv2.putText(im0,"HEAD-POSE",(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                next_txt_height += gap_txt_height
                cv2.putText(im0,"roll:"+roll_str,(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                next_txt_height += gap_txt_height
                cv2.putText(im0,"yaw:"+yaw_str,(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                next_txt_height += gap_txt_height
                cv2.putText(im0,"pitch:"+pitch_str,(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Stream results
            if view_img:
                # if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                #     break
                fps = 1.0 / (end - start)
                # display_img = img_queue.get()
                # display_fps = fps_queue.get()
                # fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
                im0 = show_fps(im0, fps)
                cv2.imshow(WINDOW_NAME, im0)
                key = cv2.waitKey(1)
                # cv2.imshow(str(p), im0)
                if key == 27:  # ESC key: quit program
                    print("")
                    print("-------------------------------")
                    print("------ See You Next Time ------")
                    print("-------------------------------")
                    print("")
                    cv2.destroyAllWindows()
                    return 0
                elif key == ord('T') or key == ord('t'):  # Toggle fullscreen
                    show_text = not show_text
                
                # elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
                #     full_scrn = not full_scrn
                #     # print(full_scrn)
                #     set_display(WINDOW_NAME, full_scrn)
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
