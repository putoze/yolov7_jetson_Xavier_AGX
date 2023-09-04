!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: JetsonYoloV7-TensorRT
echo ----------------
echo [1]: yolov3_tenrt
echo ----------------
echo [2]: None
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
echo [0]: yolov7-tiny-20230831-five-direct.pt
echo ----------------
echo [1]: yolov7-tiny-11class-894epoch.pt
echo ----------------
echo [2]: yolov7-w6-pose.pt
echo ----------------
echo -n "Press enter to start it:"

read MY_Weights

if [ $MY_Weights -eq 0 ] ; then
    Weights='yolov7-tiny-20230831-five-direct.pt'
fi 
if [ $MY_Weights -eq 1 ] ; then
    Weights='yolov7-tiny-11class-894epoch.pt'
fi 
if [ $MY_Weights -eq 2 ] ; then
    Weights='yolov7-w6-pose.pt'
fi 

echo $Weights


#============================================================================ 

echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  detect
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
echo -n "Press enter to start it:"

read MY_mode

#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「otocam  detect」
    echo ============

    python detect.py \
    --weight ../torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt 
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「Video  detect」
    echo ============

    python detect.py \
    --weight ../torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi \
    --view-img 
    #--no-trace
fi

#============================================================================ 
if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam detect L2CS」
    echo ============

    python detect_with_L2CS.py \
    --weight ../torch_yolov7_weight/$Weights \
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
    --weight ../torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi \
    --view-img
    
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
    --weight ../torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source ./test_image/frank \
    --view-img
fi

#============================================================================ 

if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「Video detect pose」
    echo ============

    python detect_pose.py \
    --weights ../torch_yolov7_weight/$Weights \
    --conf 0.05 --iou-thres 0.65 \
    --img-size 1280 \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi \
    --no-trace 

fi

#============================================================================ 

if [ $MY_mode -eq 6 ] ; then
    echo ============
    echo 「otocam detect landmark」
    echo ============

    python detect_landmark.py \
    --weight ../torch_yolov7_weight/$Weights \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt \
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


