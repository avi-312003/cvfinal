# utils.py
import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of Markers Detected: ", noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(os.path.join(path, imgPath))
        augDics[key] = imgAug
    return augDics

def loadAugVideos(path):
    myList = os.listdir(path)
    augVideos = {}
    for videoPath in myList:
        key = int(os.path.splitext(videoPath)[0])
        cap = cv2.VideoCapture(os.path.join(path, videoPath))
        augVideos[key] = cap
    return augVideos

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    matrix, _ = cv2.findHomography(pts2, pts1)
    warp_img = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
    inv_mask = cv2.bitwise_not(mask)

    imgOut = cv2.bitwise_and(img, inv_mask)
    imgOut = cv2.bitwise_or(imgOut, warp_img)

    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut
