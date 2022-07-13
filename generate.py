import csv
import cv2
import numpy as np

from datetime import datetime # para getNowFilename

import random
def getRandomParameters():
    auxP = {
        'N': random.randint(1, 6),
        'lengthwiseOffset': random.random()*2-1,
        'transversalOffset': random.random() * 2 - 1,
        'lampSpectrumThickness': random.random(),
        'starSpectrumThickness': random.random(),
        'starLampGap': random.random(),
        'blurKernellSize': random.sample([5,7,9], 1)[0],
        'kBlur': random.random()*4,
        'rotationAlpha': random.random()*1.5-0.75,
        'hFlip': random.random()>0.5,
    }
    return auxP

def getNowFilename():
    now = datetime.now()
    return now.strftime('%y%m%d')+'_'+now.strftime('%H%M%S')+'_'+now.strftime('%f')

def subLamp(lamp,floatInit,floatEnd):
    return lamp[int(floatInit*len(lamp)):int(floatEnd*len(lamp))-1]

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def displayLampSpectrum(cv2,canvas,color,lamp,x,y,w,h):
    #map lamp[0] --> [x..w+x]
    aux = []
    last = lamp[len(lamp)-1][0]
    first = lamp[0][0]
    DX = last-first
    K = w/DX
    MAX_THICK = 1
    for pair in lamp:
        aux.append(((pair[0]-first)*K,int(pair[1]*MAX_THICK)))
    yh = y + h
    # print("last: "+str(last));print("first: "+str(first));print("Dx: "+str(DX));print("x: "+str(x)); print("w: "+str(w)); print("y: " + str(y)); print("yh: " + str(yh)); print(aux)
    for pair in aux:
        xi = pair[0]
        amp = pair[1]
        cv2.line(canvas, (int(x+xi), int(y)), (int(x+xi), int(yh)), color, amp)

def displayLampSpectrumPair(cv2,canvas,white,lamp,x,y,w,h,gap): #deprecated
    displayLampSpectrum(cv2,canvas,white,lamp,x,y,w,h)
    displayLampSpectrum(cv2,canvas,white,lamp,x,y+gap,w,h)

def displayStarSpectrum(cv2, canvas, color, star, x, y, w, h):
    aux = []
    last = star[len(star) - 1][0]
    first = star[0][0]
    DX = last - first
    K = w / DX
    MAX_AMP = 255
    for pair in star:
        aux.append(((pair[0] - first) * K, int(pair[1] * MAX_AMP)))
    yh = y + h
    #print("last: "+str(last));print("first: "+str(first));print("Dx: "+str(DX));print("x: "+str(x)); print("w: "+str(w)); print("y: " + str(y)); print("yh: " + str(yh)); print(aux)
    for pair in aux:
        xi = pair[0]
        amp = pair[1]
        cv2.line(canvas, (int(x + xi), int(y)), (int(x + xi), int(yh)), amp, 1)

def displayRecord(cv2, canvas, color, lamp, star, x, y, w, h, lampSpectrumThickness, starSpectrumThickness, starLampGap, recordsGap):
    displayLampSpectrum(cv2, canvas, color, lamp, x, y + recordsGap, w, lampSpectrumThickness)
    displayStarSpectrum(cv2, canvas, color, star, x, y + recordsGap + lampSpectrumThickness + starLampGap, w, starSpectrumThickness)
    displayLampSpectrum(cv2, canvas, color, lamp, x, y + recordsGap + lampSpectrumThickness + starLampGap * 2 + starSpectrumThickness, w, lampSpectrumThickness)

def paramsTextOverlap(width,height,img, params):
    # fontScale
    fontScale = 0.6
    textLeftPad = int(width * fontScale*0.01);
    textVerticalPad = int(height * fontScale*0.05);
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Blue color in BGR
    color = 255
    # Line thickness of 2 px
    thickness = 1
    # Using cv2.putText() method
    k = 0;
    for p in params:
        # org
        org = (textLeftPad, height - textVerticalPad * (1 +k))
        k = k+1;
        image = cv2.putText(img, str(p) + ": " + str(params.get(p)), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def generateRandomImage():
    params = getRandomParameters()
    #print(params)
    # parse params to variables (?
    N = params.get('N')
    lengthwiseOffset = params.get("lengthwiseOffset")
    transversalOffset = params.get("transversalOffset")
    lampSpectrumThickness = params.get("lampSpectrumThickness")
    starSpectrumThickness = params.get("starSpectrumThickness")
    starLampGap = params.get("starLampGap")
    blurKernellSize = params.get("blurKernellSize")
    kBlur = params.get("kBlur")
    rotationAlpha = params.get("rotationAlpha")
    hFlip = params.get("hFlip")

    #canvas variables
    width = 3600
    height = 1200
    canvas = np.zeros((height, width, 1), dtype="uint8")
    white = 255;

    '''
    transversalOffset = 0           # -1..1 // 1 implica que las muestras están descentradas un recordThickness completa hacia +y, -1 hacia -y
    lampSpectrumThickness = 0.4    # 0..1
    starSpectrumThickness = 1.4     # 0..2
    starLampGap = 0.2               # 0 .. n (en lampSpectrumThickness)
    blurKernellSize = 7             # 5 faster, 9 smoother
    kBlur = 1.1                      # 0..9
    rotationAlpha = -0.3            # -1..1
    hFlip = False                   # Final Horizontall Flip: (x=-x?)
    '''

    lamp = []

    with open('./lamps/fe.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        #formato a 3 columnas: #### f int, #### f dec, 0..1 amp
        for row in reader:
            lamp.append((float(row['int f'])+int(row['dec f'])/10000,float(row['amp'])))
            #lamp.append((float(row['int f']) + int(row['dec f']) / 10000, float(str(row['amp']).replace(",", ".")))) #float amp?
            #print(row['int f']+","+row['dec f']+" "+row['amp'])
    #print(lamp)

    star = []
    for i in range(6000):
        star.append( (i,np.cos(i*i/80060)/5+0.5) )

    recordsW = width*0.9;
    recordsH = height/(N+1);
    hModule = recordsH/6;
    lampSpectrumThickness = hModule * lampSpectrumThickness
    starSpectrumThickness = hModule * starSpectrumThickness
    starLampGap = hModule * starLampGap
    recordsGap = (recordsH - 2*lampSpectrumThickness - 2*starLampGap - starSpectrumThickness)/2

    #dev simple lamp test
    # lamp = [(4.2,1),(5.2,1),(7,1)]

    # lamp = subLamp(lamp,0.5,0.8) # TODO agregar randomicidad en la escala // cropp 

    recordsX = int( (width-recordsW)/2 * (1 + lengthwiseOffset))

    for i in range(N):
        #lamp = subLamp(lamp,0,1-i/N)
        #displayRecord(cv2, canvas, color, lamp, x, y, w, h, lampSpectrumThickness, starSpectrumThickness, starLampGap, recordsGap):
        displayRecord(cv2,canvas,white,lamp,star,recordsX,i*recordsH+(recordsH/2*(transversalOffset+1)),recordsW,recordsH,lampSpectrumThickness, starSpectrumThickness, starLampGap, recordsGap)
        #displayLampSpectrumPair(cv2,canvas,white,lamp,recordsX,PAD/2+i*PAD,recordsW,recordsH,GAP)
        #displayLampSpectrumPair(cv2, canvas, white, lamp, 0, PAD / 2 + i * PAD, recordsW, recordsH, GAP)
    #cv2.imshow("Canvas", canvas)
    #cv2.waitKey(0)

    # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    #blur = cv2.bilateralFilter(canvas,5,int(width*0.05),int(width*0.01))

    img = cv2.bilateralFilter(canvas,blurKernellSize,kBlur*kBlur*100,0)
    if(hFlip):
        img = cv2.flip(img, 1)
    img = rotate_image(img,rotationAlpha)

    nameNow = getNowFilename();
    cv2.imwrite('./out/'+nameNow+'.jpg', img)

    img = paramsTextOverlap(width,height,img,params)
    cv2.imwrite('./out/'+nameNow+'_txt.jpg', img)

    # cv2.imshow("Canvas", blur); cv2.waitKey(0)

    '''
    blur = cv2.bilateralFilter(canvas,blurKernellSize,0,0)
    cv2.imwrite('./out/'+getNowFilename()+'.jpg', blur)
    '''

    #cv2.imwrite('./out/'+getNowFilename()+'_0.jpg', blur)
    #blur = cv2.bilateralFilter(canvas,5,int(width*0.05),0)
    #cv2.imwrite('./out/'+getNowFilename()+'_005.jpg', blur)
    #cv2.imwrite('./out/'+getNowFilename()+'.jpg', img)

    '''
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(canvas,-1,kernel)
    cv2.imshow("Canvas", dst)
    cv2.waitKey(0)
    '''

J = 300;
for j in range(J):
    print("generating "+str(j)+"/"+str(J))
    generateRandomImage()
