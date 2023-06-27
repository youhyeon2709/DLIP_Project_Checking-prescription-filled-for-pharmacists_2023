import torch
import cv2
from matplotlib import pyplot as plt
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
import yaml

'''
load model
'''

weight_path = 'best.pt'
model = YOLO(weight_path)

'''
class name
'''

with open('data.yaml') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    
    # Access the 'names' key and loop through the items
    cls_name = yaml_data.get('names')
    if cls_name:
        for name in cls_name:
            print(name)
    else:
        print("'names' key not found in the YAML file")

'''
video open
'''
input_file = 'day2_wide.mp4'
cap = cv2.VideoCapture(input_file)
cap.set(cv2.CAP_PROP_FPS, 60.0) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps     = int(cap.get(cv2.CAP_PROP_FPS))
user_font    = cv2.FONT_HERSHEY_COMPLEX


'''
video storage
'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = f'output_{input_file}'
frame_size = (width, height)
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

'''
load prescription
'''
prescription = 'prescription.xlsx'
df_prescription = pd.read_excel(prescription)                           # 엑셀 파일을 데이터프레임으로 읽기

'''
Define color
'''
colors = {0: (232, 168, 63),  # blue for class 0 
          1: (211,85,186),  # Purple for class 1
          }
green = (127, 232, 63)
red = (84,63,232)
black = (0,0,0)
sky = (208,224,64)
white=(255,255,255)


'''
section coordination
'''
recxyxy = { 0: (180,250,380,650,'sec1'),
            1:(410,250,610,650,'sec2'),
            2:(640,250,840,650,'sec3'),
            3:(890,250,1090,650,'sec4'),
            4:(1120,250,1320,650,'sec5'),
            5:(1370,250,1570,650,'sec6'),
            6:(180,700,380,1080,'sec7'),
            7:(410,700,610,1080,'sec8'),
            8:(640,700,840,1080,'sec9'),
            9:(890,700,1090,1080,'sec10'),                
}

alpha = 0.3

counts={ 
        'sec1':[0,0,white],
        'sec2':[0,0,white],
        'sec3':[0,0,white],
        'sec4':[0,0,white],
        'sec5':[0,0,white],
        'sec6':[0,0,white],
        'sec7':[0,0,white],
        'sec8':[0,0,white],
        'sec9':[0,0,white],
        'sec10':[0,0,white], 
}

'''
section 그리기
'''

def drawsec(image,recxyxy):
    
    overlay = image.copy()
    
    for i in range(10):
        cv2.putText(image, recxyxy[i][4], recxyxy[i][:2],cv2.FONT_ITALIC, 1, black, 2)
        cv2.rectangle(overlay, recxyxy[i][:2],recxyxy[i][2:4], counts[recxyxy[i][4]][2], -1)        #draw rectangle
          
    image = cv2.addWeighted(overlay, alpha, image,  1 - alpha, 0)                                   # Composite the rectangle to the original image as translucent
    return image

'''
Count medicine
'''

def counts_med(centerX,centerY):
    
    if (recxyxy[0][0] < centerX< recxyxy[0][2]) and (recxyxy[0][1] < centerY < recxyxy[0][3]):       #  Min X of sec1  < CenterX < Max X of sec1 & Min Y of sec1  < CenterY < Max Y of sec1
        counts['sec1'][cls] +=1
        
    elif (recxyxy[1][0] < centerX< recxyxy[1][2]) and (recxyxy[1][1] < centerY < recxyxy[1][3]):
        counts['sec2'][cls] +=1
        
    elif (recxyxy[2][0] < centerX < recxyxy[2][2]) and (recxyxy[2][1] < centerY < recxyxy[2][3]):
        counts['sec3'][cls] +=1
        
    elif (recxyxy[3][0] < centerX < recxyxy[3][2]) and (recxyxy[3][1] < centerY < recxyxy[3][3]):
        counts['sec4'][cls] +=1
        
    elif recxyxy[4][0] < centerX < recxyxy[4][2] and recxyxy[4][1] < centerY < recxyxy[4][3]:
        counts['sec5'][cls] +=1
        
    elif recxyxy[5][0] < centerX < recxyxy[5][2] and recxyxy[5][1] < centerY < recxyxy[5][3]:
        counts['sec6'][cls] +=1
        
    elif recxyxy[6][0] < centerX < recxyxy[6][2] and recxyxy[6][1] < centerY < recxyxy[6][3]:
        counts['sec7'][cls] +=1
        
    elif recxyxy[7][0] < centerX < recxyxy[7][2] and recxyxy[7][1] < centerY < recxyxy[7][3]:
        counts['sec8'][cls] +=1
        
    elif recxyxy[8][0] < centerX < recxyxy[8][2] and recxyxy[8][1] < centerY < recxyxy[8][3]:
        counts['sec9'][cls] +=1
        
    elif recxyxy[9][0] < centerX < recxyxy[9][2] and recxyxy[9][1] < centerY < recxyxy[9][3]:
        counts['sec10'][cls] +=1

'''
Define Section's color
'''
def section_color():
    
    for i in range(len(counts)):
        if counts[f'sec{i+1}'][:2]==[0,0]:      # if section is empty
            counts[f'sec{i+1}'][2] = white
            
        elif counts[f'sec{i+1}'][:2]==[1,2]:    # if Medication dispensed correctly.
            counts[f'sec{i+1}'][2] = green
            
        else :                                  # wrong
            counts[f'sec{i+1}'][2] = red
            
# Loop through the video frames
while cap.isOpened():

    start_time = time.time()
    prev_time = start_time
    
    ret, frame = cap.read()
            
    for key in counts:
        counts[key][:2] = [0, 0]                            #initialize counts dictionary
        counts[key][2] = 'white'

    if ret == True: # Run YOLOv8 inference on the frame
        
        results = model(frame, imgsz=640)
        result = results[0]
        len_result = len(result)
        
        if len_result != 0:                                 # object are decteced            
        
                for idx in range(len_result):

                    detection = result[idx]
                    box = detection.boxes.cpu().numpy()[0]
                    conf = box.conf[0]
                    
                    if conf<0.5 : continue
                    cls = int(box.cls[0])                

                    xywh    = box.xywh[0].astype(int)
                    centerX = xywh[0]
                    centerY = xywh[1]
                    area    = xywh[2] * xywh[3]

                
                    color = colors[cls]
                    
                    counts_med(centerX,centerY)               #count pills in section
                    
                    r = box.xyxy[0].astype(int) # box
                    
                    class_id = result.names[cls]
                    conf = round(conf.item(), 2)

                    cv2.rectangle(frame, r[:2], r[2:], color, 2)
                    cv2.putText(frame, f'{cls_name[cls]}:{conf:.2f}', (r[0]-5, r[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
        section_color()                                                 #determine section's color
        frame = drawsec(frame,recxyxy)                                  #draw section
        
        '''
        calculate FPS
        '''

        diff_time = time.time() - prev_time
        
        if diff_time > 0:
            fps = 1 / diff_time

        cv2.putText(frame, f'FPS : {fps:.2f}', (20, 40), user_font, 1, green, 2)
        
        out.write(frame)
        
        resized_image = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)                     # 이미지 크기 조정
        cv2.imshow("mask", resized_image)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q')   :   break

    else:
        print("Vidieo is empty")
        break

cap.release()
cv2.destroyAllWindows()

