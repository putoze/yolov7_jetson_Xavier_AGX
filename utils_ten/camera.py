"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  In
addition, this Camera class is further extended to take a video
file or an image file as input.
"""


import logging
import threading
import subprocess

import numpy as np
import cv2


# The following flag ise used to control whether to use a GStreamer
# pipeline to open USB webcam source.  If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery. i.e. relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True


def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--image', type=str, default=None,
                        help='image file name, e.g. dog.jpg')
    parser.add_argument('--video', type=str, default=None,
                        help='video file name, e.g. traffic.mp4')
    parser.add_argument('--video_looping', action='store_true',
                        help='loop around the video file [False]')
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')
    parser.add_argument('--usb', type=int, default=None,
                        help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--gstr', type=int, default=None,
                        help='GStreamer string [None]')
    parser.add_argument('--onboard', type=int, default=None,
                        help='Jetson onboard camera [None]')
    parser.add_argument('--copy_frame', action='store_true',
                        help=('copy video frame internally [False]'))
    parser.add_argument('--do_resize', action='store_true',
                        help=('resize image/video [False]'))
    parser.add_argument('--width', type=int, default=640,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=480,
                        help='image height [480]')
    # self add
    parser.add_argument('--save_img', type=str, default='./save_img/save_img',
                        help='determine save img path')
    parser.add_argument('--save_record', type=str, default='./save_img/save_record',
                        help='determine save video path')
    
    return parser


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_gstr(gstr, width, height):
    gst_str = ('v4l2src device=/dev/video{} ! '
                'video/x-raw, width=1280, height=722, format=GRAY16_LE ! '
                'autovideoconvert ! appsink').format(gstr)
    print(gst_str)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False


class Camera():
    """Camera class which supports reading images from theses video sources:

    1. Image (jpg, png, etc.) file, repeating indefinitely
    2. Video file
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    """

    def __init__(self, args, stride=32, img_size=640):
        self.args = args
        self.is_opened = False
        self.video_file = ''
        self.video_looping = args.video_looping
        self.thread_running = False
        self.img_handle = None
        self.copy_frame = args.copy_frame
        self.do_resize = args.do_resize
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        self.stride = stride
        self.img_size = img_size
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args
        if a.image:
            logging.info('Camera: using a image file %s' % a.image)
            self.cap = 'image'
            self.img_handle = cv2.imread(a.image)
            if self.img_handle is not None:
                if self.do_resize:
                    self.img_handle = cv2.resize(
                        self.img_handle, (a.width, a.height))
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_handle.shape
        elif a.video:
            logging.info('Camera: using a video file %s' % a.video)
            self.video_file = a.video
            self.cap = cv2.VideoCapture(a.video)
            self._start()
        elif a.rtsp:
            logging.info('Camera: using RTSP stream %s' % a.rtsp)
            self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
            self._start()
        elif a.usb is not None:
            logging.info('Camera: using USB webcam /dev/video%d' % a.usb)
            self.cap = open_cam_usb(a.usb, a.width, a.height)
            self._start()
        elif a.gstr is not None:
            logging.info('Camera: using GStreamer string "%s"' % a.gstr)
            self.cap = open_cam_gstr(a.gstr, a.width, a.height)
            self._start()
        elif a.onboard is not None:
            logging.info('Camera: using Jetson onboard camera')
            self.cap = open_cam_onboard(a.width, a.height)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()

        # If input is gray image, transfer it into BGR 3 dim format
        if len(self.img_handle.shape) < 3:
            self.img_handle = cv2.cvtColor(self.img_handle, cv2.COLOR_GRAY2BGR)

        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
        if self.video_file:
            if not self.do_resize:
                self.img_height, self.img_width, _ = self.img_handle.shape
        else:
            self.img_height, self.img_width, _ = self.img_handle.shape
            # start the child thread if not using a video file source
            # i.e. rtsp, usb or onboard
            assert not self.thread_running
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()
            # check for common shapes

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            #self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None

        if self.video_file:
            _, img = self.cap.read()
            if img is None:
                logging.info('Camera: reaching end of video file')
                if self.video_looping:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_file)
                _, img = self.cap.read()
            if img is not None and self.do_resize:
                img = cv2.resize(img, (self.img_width, self.img_height))
            return img
        elif self.cap == 'image':
            return np.copy(self.img_handle)
        else:
            if self.copy_frame:
                return self.img_handle.copy()
            else:
                return self.img_handle

    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()