!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: JetsonYoloV7-TensorRT
echo ----------------
echo [1]: yolov3_tenrt
echo ----------------
echo [n]: None
echo ----------------
echo -n "Press enter to start it:"

read ENV_Set

#============================================================================
if [ $ENV_Set -eq 0 ] ; then
    source activate
    conda activate JetsonYoloV7-TensorRT   

    echo ============
    echo 「Success Enter JetsonYoloV7-TensorRT」
    echo ============ 
fi

#============================================================================

if [ $ENV_Set -eq 1 ] ; then
    source activate
    conda activate yolov3_tenrt

    echo ============
    echo 「Success Enter yolov3_tenrt」
    echo ============
fi

#============================================================================ 

echo ""
echo "Hello, choose the weight you want it~"
echo [0]: yolov7-tiny-20230831-five-direct-1100epoch.pt
echo ----------------
echo [1]: yolov7-tiny-20230812-11class-894epoch.pt
echo ----------------
echo [2]: yolov7-w6-pose.pt
echo ----------------
echo [3]: yolov8-lite-t.onnx
echo ----------------
echo [4]: yolov8-lite-s.onnx
echo ----------------
echo [5]: yolov7-tiny-20230831-five-direct-2200epoch.pt
echo ----------------
echo [6]: yolov7-tiny-9cs-70cm-20230916-600epoch.pt
echo ----------------
echo [7]: yolov7-tiny-5cs-50cm-20230916-600epoch.pt
echo ----------------
echo [8]: yolov7-tiny-9cs-20230916-50_70cm-600epoch.pt
echo ----------------
echo [n]: None
echo -n "Press enter to start it:"


read MY_Weights

if [ $MY_Weights -eq 0 ] ; then
    Weights='yolov7-tiny-20230831-five-direct-1100epoch.pt'
fi 
if [ $MY_Weights -eq 1 ] ; then
    Weights='yolov7-tiny-20230812-11class-894epoch.pt'
fi 
if [ $MY_Weights -eq 2 ] ; then
    Weights='yolov7-w6-pose.pt'
fi 
if [ $MY_Weights -eq 3 ] ; then
    Weights='yolov8-lite-t.onnx'
fi 
if [ $MY_Weights -eq 4 ] ; then
    Weights='yolov8-lite-s.onnx'
fi 
if [ $MY_Weights -eq 5 ] ; then
    Weights='yolov7-tiny-20230831-five-direct-2200epoch.pt'
fi 
if [ $MY_Weights -eq 6 ] ; then
    Weights='yolov7-tiny-9cs-70cm-20230916-600epoch.pt'
fi
if [ $MY_Weights -eq 7 ] ; then
    Weights='yolov7-tiny-5cs-50cm-20230916-600epoch.pt'
fi
if [ $MY_Weights -eq 8 ] ; then
    Weights='yolov7-tiny-9cs-20230916-50_70cm-600epoch.pt'
fi


echo $Weights


#============================================================================ 

echo ""
echo "Hello, choose the mode you want it~"
echo ------ Demo ------
echo [0]: img + save_txt  detect
echo ----------------
echo [1]: Video  detect
echo ----------------
echo [2]: otocam  detect + L2CS
echo ----------------
echo [3]: Video  detect + L2CS
echo ----------------
echo [4]: Image  detect + L2CS
echo ----------------
echo [5]: Video  detect + pose
echo ----------------
echo [6]: otocam  detect + landmark
echo ----------------
echo [7]: otocam  yolov8-face-demo
echo ----------------
echo [8]: otocam  detect + 6D headpose
echo ----------------
echo [9]: video  detect + 6D headpose
echo ----------------
echo [10]: otocam  detect + 6D headpose + gazeml
echo ----------------
echo [11]: video  detect + 6D headpose + gazeml
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「img + save_txt  detect」
    echo ============

    python detect.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/img_new/all/ \
    --save-txt
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「Video  detect」
    echo ============

    python detect.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output30.avi \
    --no-trace
    # --source /media/joe/Xavierssd/20230816_window_video/20230318205957.mp4

fi

#============================================================================ 

if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam detect L2CS」
    echo ============

    python detect_with_L2CS.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt

fi

#============================================================================ 

if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「Video detect L2CS」
    echo ============

    python detect_with_L2CS.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --no-trace
fi

#============================================================================ 

if [ $MY_mode -eq 4 ] ; then
    echo ============
    echo 「Image detect L2CS」
    echo ============

    # for filelist in ./test_image/frank/*.png;do

    # python detect_with_L2CS.py \
    # --weight ../torch_yolov7_weight/$Weights \
    # --conf 0.5 \
    # --img-size 640 \
    # --source $filelist \
    # --view-img
    # done
    python detect_with_L2CS.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source ./test_image/frank \
    --view-img
fi

#============================================================================ 

if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「otocam detect pose」
    echo ============

    python detect_pose.py \
    --weights ../weights/torch_yolov7_weight/$Weights \
    --conf 0.05 --iou-thres 0.65 \
    --img-size 1280 \
    --source cam.txt \
    --no-trace 

fi

#============================================================================ 

if [ $MY_mode -eq 6 ] ; then
    echo ============
    echo 「otocam detect landmark」
    echo ============

    python detect_landmark.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt \
    --no-trace 

fi

#============================================================================ 


if [ $MY_mode -eq 7 ] ; then
    echo ============
    echo 「otocam  yolov8-face-demo」
    echo ============

    python3 yolov8-face-demo.py \
    --modelpath ../weights/yolov8-face-weights/$Weights \
    --gstr 1 --save_img ./save_img/save_img \
    --save_record ./save_img/save_record 
    
fi


#============================================================================ 


if [ $MY_mode -eq 8 ] ; then
    echo ============
    echo 「otocam  detect_with_6D」
    echo ============

    python detect_with_6D.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt \
    --no-trace 
    
fi

#============================================================================ 


if [ $MY_mode -eq 9 ] ; then
    echo ============
    echo 「video  detect_with_6D」
    echo ============

    python detect_with_6D.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --no-trace \
    --view-img \
    --nosave
    
fi

#============================================================================ 

if [ $MY_mode -eq 10 ] ; then
    echo ============
    echo 「otocam  detect_with_6D_gazeml」
    echo ============

    python detect_with_6D_gazeml.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt \
    --no-trace 
    
fi

#============================================================================ 

if [ $MY_mode -eq 11 ] ; then
    echo ============
    echo 「Video detect_with_6D_gazeml」
    echo ============

    python detect_with_6D_gazeml.py \
    --weight ../weights/torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --no-trace

fi


#============================================================================ 

# if [ $MY_mode -eq] ; then
#     echo ============
#     echo 「export to onnx」
#     echo ============

#     python export.py \
#     --weight ../torch_yolov7_weight/yolov7_tiny_coco/yolov7-tiny.pt \
#     --grid \
#     --end2end --simplify \
#     --topk-all 100 \
#     --iou-thres 0.65 \
#     --conf-thres 0.35 \
#     --img-size 640 \
#     --max-wh 640 
#     #--view-img \
#     #--no-trace
# fi

#============================================================================ 

# if [ $MY_mode -eq] ; then
#     echo ============
#     echo 「export to trt 」
#     echo ============

#     python ../tensorrt-python/export.py \
#     -o torch_yolov7_weight/yolov7_tiny_coco/yolov7-tiny.onnx \
#     -e torch_yolov7_weight/yolov7_tiny_coco/yolov7-tiny.trt \
#     -p fp16
#     #--view-img \
#     #--no-trace
# fi

#============================================================================ End
echo [===YOLO===] ok!


