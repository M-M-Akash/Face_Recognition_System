import paddlehub as hub
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
import cv2
import json
from src.VideoStream import VideoStream

face_detector = hub.Module(name="pyramidbox_lite_mobile")
parser = face.parser()
args = parser.parse_args()


args.use_gpu = False
args.enable_mkldnn = True
args.cpu_threads = 4
args.det = False
args.rec = True
args.rec_thresh = 0.45
args.index = "Dataset/index.bin"
args.rec_model = "Models/mobileface_v1.0_infer"
recognizer = face.InsightFace(args)



def detect_face(image):
    result = face_detector.face_detection(images=[image], use_gpu=False)
    box_list = result[0]['data']
    return box_list
def recognize_face(image, box_list):
    img = image[:, :, ::-1]
    res = list(recognizer.predict(img, box_list, print_info=True))
    box_list = res[0]['box_list']
    labels = res[0]['labels']
    return box_list, labels

def draw_boundary_boxes(image, box_list, labels):
    
    for box,label in zip(box_list,labels):
        score = "{:.2f}".format(box['confidence'])
        x_min, y_min, x_max, y_max = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Put the label text near the box
        label = label+str(score)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def detection_video_stream(camera_urls):
    # Create VideoStream instances for each camera
    cameras = [VideoStream(url) for url in camera_urls]

    # Start reading frames from each camera
    for camera in cameras:
        camera.start()

    # Display frames from each camera in separate windows
    while cameras:
        for i, camera in enumerate(cameras):
            frame = camera.read()
            
            if frame is not None:
                # resize frame to match the input size of the model
                resized_frame = cv2.resize(frame, (640,480), interpolation= cv2.INTER_LINEAR)
                box_list = detect_face(resized_frame)
                box_list, labels = recognize_face(resized_frame, box_list)
                draw_boundary_boxes(resized_frame, box_list, labels)
                cv2.imshow(f"Camera {i+1}", resized_frame)
            else:
                print(f"Camera {cameras.index(camera)+1} disconnected")
                camera.stop()
                cv2.destroyAllWindows()
                cameras.remove(camera)
                

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Stop reading frames from each camera
    for camera in cameras:
        camera.stop()

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    with open('camera_urls.json', 'r') as f:
        camera_urls = json.load(f)
    detection_video_stream(camera_urls)


