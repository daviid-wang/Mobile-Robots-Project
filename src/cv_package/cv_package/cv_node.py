# import cv2

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v') 
# writer = cv2.VideoWriter("roborecord.mp4", fourcc, 30.0, (1280,720))
# is_recording = False

# while True:
#     return_value, frame = cap.read()
#     if return_value:
#         cv2.imshow("video", frame)
#         if is_recording:
#             writer.write(frame) #writing frame into the file
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break #quit
#     elif key == ord('r'):
#         is_recording = not is_recording
#         print(f"Recording {is_recording}")
        
# cap.release()
# writer.release()
# cv2.destroyAllWindows()

from depthai_sdk import OakCamera
from depthai_sdk.record import RecordType

# with OakCamera() as oak:
#     color = oak.create_camera('color', resolution='1080p')
#     oak.visualize([color])
#     oak.start(blocking=False)

#     while oak.running():
#         oak.poll()
#         # this code is executed while the pipeline is running
def print_num_objects(packet):
    print(f'Number of objects detected: {len(packet.detections)}')
    
with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    # List of models that are supported out-of-the-box by the SDK:
    # https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
    yolo = oak.create_nn('yolov6n_coco_640x640', input=color)

    oak.record([color, yolo], path='./records', record_type=RecordType.VIDEO)
    oak.start(blocking=True)