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

# [./darknet] --> [./home/lab716/Desktop/Rain/darknet/darknet]
#============================================================================ 

echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  detect
echo ----------------
echo [1]: Video  detect
echo ----------------
echo [2]: otocam trt_yolo
echo ----------------
echo [3]: otocam  detect + L2CS
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「otocam detect torch_yolov7_weight/yolov7-tiny-20230831-five-direct.pt」
    echo ============

    python detect.py \
    --weight ../torch_yolov7_weight/yolov7-tiny-20230831-five-direct/yolov7-tiny-20230831-five-direct.pt \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt 
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「Video torch_yolov7_weight/yolov7-custom_v3/best.pt」
    echo ============

    python detect.py \
    --weight ../torch_yolov7_weight/yolov7-custom_v3/best.pt \
    --conf 0.5 \
    --img-size 640 \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi
    #--view-img \
    #--no-trace
fi

#============================================================================ 
if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam trt_yolo torch_yolov7_weight/yolov7-tiny-20230831-five-direct.pt」
    echo ============

    python trt_yolo.py \
    --weight ../torch_yolov7_weight/yolov7-tiny-20230831-five-direct/yolov7-tiny-20230831-five-direct.pt \
    --conf 0.5 \
    --img-size 640 \
    --gstr 1 --save_img ./save_img/save_img \
    --save_record ./save_img/save_record 
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
fi

#============================================================================ 



 if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「otocam detect L2CS torch_yolov7_weight/yolov7-tiny-20230831-five-direct.pt」
    echo ============

    python detect_with_L2CS.py \
    --weight ../torch_yolov7_weight/yolov7-tiny-20230831-five-direct/yolov7-tiny-20230831-five-direct.pt \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt \
    --snapshot models/Gaze360/L2CSNet_gaze360.pkl
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
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


