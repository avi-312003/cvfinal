# marker_app/views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import io
from django.http import StreamingHttpResponse
from .utils import findArucoMarkers, augmentAruco, loadAugImages, loadAugVideos



def home(request):
    return render(request, 'home.html')


def load_aug_images(path):
    """
    Load augmented images from the specified path.
    Returns a dictionary with marker IDs as keys and corresponding images as values.
    """
    my_list = os.listdir(path)
    aug_dict = {}
    for img_path in my_list:
        key = int(os.path.splitext(img_path)[0])
        img_aug = cv2.imread(os.path.join(path, img_path))
        aug_dict[key] = img_aug
    return aug_dict

def find_aruco_markers(img, marker_size=6, total_markers=250, draw=True):
    """
    Find Aruco markers in the given image.
    Returns bounding boxes and IDs of detected markers.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_param)

    if draw:
        # Ensure img is not None before calling drawDetected
        if img is not None:
            aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids

def augment_aruco(bbox, id, img, img_aug, draw_id=True):
    """
    Augment Aruco marker in the image with the corresponding augmented image.
    """
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = img_aug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    matrix, _ = cv2.findHomography(pts2, pts1)
    img_out = cv2.warpPerspective(img_aug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    img_out = img + img_out

    if draw_id:
        cv2.putText(img_out, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return img_out

def upload_image(request):
    result_image_url = None
    if request.method == 'POST' and 'image' in request.FILES:
        image = request.FILES['image']

        try:
            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return HttpResponse(f"Error decoding image: {str(e)}", status=500)

        if img is None:
            return HttpResponse("Error: Unable to decode the image.", status=500)

        aug_images = load_aug_images(os.path.join(settings.BASE_DIR, 'Markers'))
        aruco_found = find_aruco_markers(img)

        if len(aruco_found[0]) != 0:
            for bbox, marker_id in zip(aruco_found[0], aruco_found[1]):
                if int(marker_id) in aug_images.keys():
                    img = augment_aruco(bbox, marker_id, img, aug_images[int(marker_id)])

        _, img_encoded = cv2.imencode('.png', img)

        # Use io.BytesIO to create a file-like object
        img_bytes = io.BytesIO(img_encoded.tobytes())
        img_bytes.seek(0)

        # Create an HttpResponse with the image content type
        response = HttpResponse(content_type="image/png")
        response.write(img_bytes.read())
        return response

    return render(request, 'upload_image.html')

def generate_frames(cap, aug_images):
    while True:
        success, img = cap.read()
        if not success:
            break

        aruco_found = find_aruco_markers(img)

        if len(aruco_found[0]) != 0:
            for bbox, marker_id in zip(aruco_found[0], aruco_found[1]):
                if int(marker_id) in aug_images.keys():
                    img = augment_aruco(bbox, marker_id, img, aug_images[int(marker_id)])

        _, img_encoded = cv2.imencode('.jpg', img)

        img_bytes = img_encoded.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKERS_PATH = os.path.join(BASE_DIR, 'Markers')
VIDEO_PATH = os.path.join(BASE_DIR, 'Videos')

def live_camera(request):
    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield HttpResponse("Error: Webcam not found. Make sure it is connected and accessible.")
            return

        # Load augmented images and videos
        augDics = loadAugImages(MARKERS_PATH)
        augVideos = loadAugVideos(VIDEO_PATH)

        while True:
            success, img = cap.read()

            if not success:
                yield HttpResponse("Failed to capture frame from the webcam.")
                return

            arucoFound = findArucoMarkers(img)

            if len(arucoFound[0]) != 0:
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                    if int(id) in augDics.keys():
                        img = augmentAruco(bbox, id, img, augDics[int(id)], drawId=False)
                    if int(id) in augVideos.keys():
                        video_capture = augVideos[int(id)]
                        ret, frame = video_capture.read()
                        if ret:
                            img = augmentAruco(bbox, id, img, frame, drawId=False)
                        else:
                            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning

            _, img_encoded = cv2.imencode('.jpg', img)
            img_encoded = img_encoded.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded + b'\r\n')

    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace;boundary=frame")


def upload_video(request):
    if request.method == 'POST' and 'video' in request.FILES:
        video = request.FILES['video']
        video_path = os.path.join('media', 'uploaded_video.mp4')
        
        # Save the uploaded video
        with open(video_path, 'wb') as f:
            f.write(video.read())

        augDics = loadAugImages("Markers")
        augVideos = loadAugVideos("Videos")
        
        def generate_video_frames():
            cap = cv2.VideoCapture(video_path)

            while True:
                success, img = cap.read()
                if not success:
                    break

                arucoFound = findArucoMarkers(img)

                if len(arucoFound[0]) != 0:
                    for bbox, id in zip(arucoFound[0], arucoFound[1]):
                        if int(id) in augDics.keys():
                            img = augmentAruco(bbox, id, img, augDics[int(id)], drawId=False)
                        if int(id) in augVideos.keys():
                            video_capture = augVideos[int(id)]
                            ret, frame = video_capture.read()
                            if ret:
                                img = augmentAruco(bbox, id, img, frame, drawId=False)
                            else:
                                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning

                _, img_encoded = cv2.imencode('.jpg', img)
                img_bytes = img_encoded.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')

        response = StreamingHttpResponse(generate_video_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
        return response

    return render(request, 'upload_video.html')