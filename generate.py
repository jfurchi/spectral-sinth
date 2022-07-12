import LampSpectrum.py
import csv
import numpy as np
import cv2

lamp = []

with open('./lamps/fe_.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    #spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        lamp.append((float(row['int f'])+int(row['dec f'])/10000,float(row['amp'])))
        print(row['int f']+","+row['dec f']+" "+row['amp'])
print(lamp)

canvas = np.zeros((600, 900, 1), dtype="uint8")

width = canvas.shape[1];
height = canvas.shape[0];

white = 255;
cv2.line(canvas, (0, 0), (width, height), white)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

def displayLampSpectrum(lamp):
    for pair in lamp:
        cv2.line(canvas, (0, 0), (width, height), white)

    cv2.waitKey(0)