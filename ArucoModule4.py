import cv2
import cv2.aruco as aruco
import numpy as np
import os
import streamlit as st
import tempfile

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.augDics = loadAugImages("Markers")
        self.augVideos = loadAugVideos("Videos")
        super(VideoTransformer, self).__init__()

    def transform(self, frame):
        arucoFound = findArucoMarkers(frame, draw=False)

        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in self.augDics.keys():
                    frame = augmentAruco(bbox, id, frame, self.augDics[int(id)], drawId=False)

        return frame

# def get_image_download_link(image_encoded, filename):
#     href = f'<a href="data:image/png;base64,{image_encoded}" download="{filename}.png">Download {filename}</a>'
#     return href


def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of Markers Detected: ", noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def loadAugVideos(path):
    """
    Load video files for each marker ID.
    :param path: Folder where video files are stored.
    :return: Dictionary with key as the ID and values as VideoCapture objects.
    """
    myList = os.listdir(path)
    augVideos = {}
    for videoPath in myList:
        key = int(os.path.splitext(videoPath)[0])
        cap = cv2.VideoCapture(f'{path}/{videoPath}')
        augVideos[key] = cap
    return augVideos

augDics = loadAugImages("Markers")
augVideos = loadAugVideos("Videos")  # Load videos for augmentation

def findArucoMarkers(img,markerSize=6, totalMarkers=250, draw=True):
    """
    :param image in which to find the aruco makers
    :param markersize : the size of the markers
    :param totalMarkers : total number of markers that compose the dictionary
    :param draw : flag to draw bbox around markers detected
    : return : bounding boxexs and id numbers of markers detected

    """
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs,ids, rejected = aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)

    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    Augment the ArUco marker with an image or video while preserving the background.
    :param bbox: Four corner points of the marker's bounding box.
    :param id: Marker ID.
    :param img: Image on which to draw.
    :param imgAug: The image or video to overlay on the marker.
    :param drawId: Flag to display the marker ID.
    :return: Image with the augmented image or video.
    """
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    matrix, _ = cv2.findHomography(pts2, pts1)
    warp_img = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Create a mask to blend the augmented content with the original image
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
    inv_mask = cv2.bitwise_not(mask)

    imgOut = cv2.bitwise_and(img, inv_mask)
    imgOut = cv2.bitwise_or(imgOut, warp_img)

    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut

def main():
    st.title("ArUco Marker Augmentation")
    st.sidebar.header("Choose an Option")

    option = st.sidebar.radio("Select an option:", ("Upload Image", "Live Cam","Upload Video"))

    if option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
            arucoFound = findArucoMarkers(img)
            img_copy = img.copy()

            if len(arucoFound[0]) != 0:
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                    if int(id) in augDics.keys():
                        img_copy = augmentAruco(bbox, id, img_copy, augDics[int(id)])
                    elif int(id) in augVideos.keys():
                        video_capture = augVideos[int(id)]
                        ret, frame = video_capture.read()
                        if ret:
                            img_copy = augmentAruco(bbox, id, img_copy, frame, drawId=False)
                        else:
                            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            st.image(img_copy, channels="BGR")
    elif option == "Live Cam":
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")

        cap = None  # Initialize cap as None

        if start_button:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Webcam not found. Make sure it is connected and accessible.")
            else:
                st.success("Webcam found. Camera is started.")

        if stop_button:
            if cap is not None:
                cap.release()
                st.warning("Camera stopped.")
            cap = None  # Set cap to None to indicate that the camera is not open

        if cap is not None:
            video_element = st.empty()  # Create a placeholder for the video element

            while True:
                success, img = cap.read()

                if not success:
                    st.warning("Failed to capture frame from the webcam.")
                    continue

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

                video_element.image(img, channels="BGR", use_column_width=True)

    elif option == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
        video_element = st.empty()  # Create a placeholder for the video element

        if uploaded_video is not None:
            filename = uploaded_video.name
            file_extension = filename.split('.')[-1].lower()

            if file_extension in ["mp4", "avi"]:
                video_data = uploaded_video.read()

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                    temp_file.write(video_data)
                    temp_file.seek(0)  # Move the file pointer to the beginning

                cap = None  # Initialize cap as None

                # Add "Start Video" and "Stop Video" buttons
                start_video = st.button("Start Video")
                stop_video = st.button("Stop Video")

                while True:
                    if start_video and cap is None:
                        cap = cv2.VideoCapture(temp_file.name)  # Initialize cap when starting the video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the beginning
                        start_video = False

                    if cap is not None:
                        success, img = cap.read()

                        if not success:
                            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                                st.warning("End of video. Restarting...")
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
                            else:
                                st.warning("Failed to capture frame.")
                            continue

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

                        video_element.image(img, channels="BGR", use_column_width=True)

                        if stop_video:
                            cap.release()  # Release the video capture object
                            break  # Exit the loop if "Stop Video" button is clicked

                # Clean up the temporary file
                temp_file.close()
            else:
                st.error("Unsupported video format. Please upload an 'mp4' or 'avi' video.")
        else:
            st.warning("Please select an option.")




if __name__ == "__main__":
    main()