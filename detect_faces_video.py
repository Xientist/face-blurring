#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:07:27 2020

@author: julien
"""

# importation des packages nécessaires
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# on précise les configurations
# model: le modèle caffe pré-entrainé de OpenCV (réseau de neurones)
# prototxt: fichier de déploiement du modèle caffe
# confidence: niveau de confiance minimal pour la détection de visage
# pixmax: longueur et largeur maximum de la pixélisation
# pixdefault: longueur et largeur par défaut de la pixélisation
args = {
        "model": "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        "prototxt": "deploy_lowres.prototxt",
        "confidence": 0.5,
        "pixmax": 16,
        "pixdefault": 8
    }

# fonction de pixélisation dans une zone
def anonymize_face_pixelate(image, blocks=3):
    # on divise la source en NxN blocs
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # on boucle sur les sous-blocs créés précédemments
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # on récupère les coordonnées x et y du bloc courant
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # on extrait le sous-bloc de l'image
            roi = image[startY:endY, startX:endX]
            # on effectue la moyenne de couleur de tout les pixels 
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            # on dessine un rectangle de la couleur moyenne obtenue
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # on retourne l'image pixélisée
    return image

# chargement du modèle caffe
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# on initialise le flux vidéo de Imutils
print("[INFO] starting video stream...")
vs = VideoStream(src=0)
vs.start()
time.sleep(2.0)

# on utilise l'affichage sans interface graphique QT
cv2.namedWindow("Frame", cv2.WINDOW_GUI_NORMAL | 
                cv2.WINDOW_AUTOSIZE | 
                cv2.WINDOW_KEEPRATIO)

# on initialise le nombre de blocs courants pour la pixélisation
current = { "blocks": args["pixdefault"] }

# fonction qui change le niveau de pixélisation courant
def on_trackbar(val):
    current["blocks"] = val if val>0 else 1
    
# on crée un slider pour changer dynamiquement la pixélisation
# elle prends en paramètre la fonction précédente
cv2.createTrackbar("degré de pixélisation",
                   "Frame", 
                   args["pixdefault"],
                   args["pixmax"],
                   on_trackbar)

# boucle principale qui traite la sortie courante du flux en continu
while True:
    # on récupère l'image depuis le flux de la caméra
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 
    # l'image est redimensionnée a 300x300 pixels et normalisée
    # pour être mise en entré dans le réseau de neurones 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # on injecte l'image formattée pour le réseau
    # on peut ainsi récupérer les potentiels visages détectés 
    net.setInput(blob)
    detections = net.forward()
    
    # on boucle sur les détections obtenues
    for i in range(0, detections.shape[2]):
        # on récupère le niveau de confiance de la détection courante
        confidence = detections[0, 0, i, 2]
        # on l'ignore si la confiance n'est pas suffisament élevée
        if confidence < args["confidence"]:
            continue
        # on calcul les positions des quatres coins du visage
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # on extrait le visage de l'image
        face = frame[startY:endY, startX:endX]
        # enfin, on applique la pixélisation
        face = anonymize_face_pixelate(face, current["blocks"])
        frame[startY:endY, startX:endX] = face
        
    # on affiche le résultat
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # on quitte le programme avec la touche 'q' ou en appuyant 
    # sur la croix de la fenêtre
    if key == ord("q"):
        break
    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break
    
# on libère les fenêtres et le flux vidéo
cv2.destroyAllWindows()
vs.stream.stream.release()