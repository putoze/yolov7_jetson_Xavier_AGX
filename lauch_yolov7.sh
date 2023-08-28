#!/bin/sh
echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  yolov7-best
echo ----------------
echo [1]: Video  yolov7-best
echo ----------------
echo [2]: export to onnx
echo ----------------
echo [3]: export to trt
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

# [./darknet] --> [./home/lab716/Desktop/Rain/darknet/darknet]
#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「otocam torch_yolov7_weight/yolov7-custom_v3/best.pt」
    echo ============

    python detect.py \
    --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    --conf 0.5 \
    --img-size 640 \
    --source cam.txt
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「Video torch_yolov7_weight/yolov7-custom_v3/best.pt」
    echo ============

    python detect.py \
    --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    --conf 0.5 \
    --img-size 640 \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi
    #--view-img \
    #--no-trace
fi

#============================================================================ 

if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「export to onnx」
    echo ============

    python export.py \
    --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    --grid \
    --end2end --simplify \
    --topk-all 100 \
    --iou-thres 0.65 \
    --conf-thres 0.35 \
    --img-size 640 \
    --max-wh 640 
    #--view-img \
    #--no-trace
fi

#============================================================================ 

if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「export to trt 」
    echo ============

    python ./tensorrt-python/export.py \
    -o runs/train/yolov7-custom_v2/weights/best.onnx \
    -e runs/train/yolov7-custom_v2/weights/best-nms.trt \
    -p fp16
    #--view-img \
    #--no-trace
fi

#============================================================================ 

if [ $MY_mode -eq 4 ] ; then
    echo ============
    echo 「otocam mtcnn」
    echo ============
    python3 trt_mtcnn.py --gstr 1

fi

#============================================================================ 

if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「Video tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker \
    --video ./../tensor_test.avi -c 5 -t 0.8 #--width 1280 --height 722

fi

#============================================================================ 

if [ $MY_mode -eq 6 ] ; then
    echo ============
    echo 「map tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 eval_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker 

fi

#============================================================================ End
echo [===YOLO===] ok!


