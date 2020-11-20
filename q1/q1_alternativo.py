#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS


# Este codigo pode ser visto rodando em: https://youtu.be/USeRmzNlNxE


from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
model = "../../robot20/ros/exemplos_python/scripts/MobileNetSSD_deploy.caffemodel"
proto = "../../robot20/ros/exemplos_python/scripts/MobileNetSSD_deploy.prototxt.txt"
# video = "../../robot20/media/dogs_table.mp4"
# video = "../../robot20/media/dogs_table_black.mp4"
# video = "../../robot20/media/animacao_bulldogs.mp4"
# video = "../../robot20/media/animacao_bulldogs2.mp4"
video = "../../robot20/media/dogs_chairs.mp4"

def check_exists_size(name, size):
    """
        Função para diagnosticar se os arquivos estão com problemas
    """
    if os.path.isfile(name):
        stat = os.stat(name)
        print("Informações do arquivo ", name, "\n", stat)
        if stat.st_size !=size:
            print("Tamanho errado para o arquivo ", name, " Abortando ")
            mensagem_falta_arquivos()
            sys.exit(0)
    else:
        print("Arquivo ", name, " não encontrado. Abortando!")
        mensagem_falta_arquivos()
        sys.exit(0)

def mensagem_falta_arquivos():
    msg = """
    Tente apagar os arquivos em robot20/ros/exemplos_python/scripts:
         MobileNetSSD_deploy.prototxt.txt
         MobileNetSSD_deploy.caffemodel
    Depois
        No diretório robot20/ros/exemplos_python/scripts fazer:
        git checkout MobileNetSSD_deploy.prototxt.txt
        Depois ainda: 
        git lfs pull 

        No diretório No diretório robot20/media

        Fazer:
        git lfs pull

        Ou então baixe os arquivos manualmente nos links:
        https://github.com/Insper/robot20/tree/master/ros/exemplos_python/scripts
        e
        https://github.com/Insper/robot20/tree/master/media
    """
    print(msg)

def detect(frame):
    """
        Recebe - uma imagem colorida
        Devolve: objeto encontrado
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

def text_cv(bgr, pos, text, fontscale=1, thickness=1):
    # Tirado do gabarito da P1 de 2020.1
    font = cv2.FONT_HERSHEY_SIMPLEX    
    color=(255,255,255)    
    mensagem = "{}".format(text)
    cv2.putText(bgr, mensagem, pos, font, fontscale, color, thickness, cv2.LINE_AA) 
    
def check_magenta_0(frame, p1, p2):
    inicio = p1
    fim = p2

    x = 0
    y =  1

    #parte = frame

    parte = frame[inicio[y]:fim[y], inicio[x]:fim[x]]

    if parte.shape[0]>1 and parte.shape[1] > 1: 

        hsv = cv2.cvtColor(parte, cv2.COLOR_BGR2HSV)
        mini = np.array([135, 25,25])#, dtype=np.uint8)
        maxi = np.array([175, 255,255])#, dtype=np.uint8)
        mask = cv2.inRange(hsv, mini , maxi)

        cv2.imshow('Mascara', mask)

        cont = 0

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 255:
                    cont=cont+1

        text_cv(parte, (1,45), '{}'.format(cont))
        cv2.imshow("Slice", parte)

        if cont > 1000: 
            return True



    return False


def check_magenta(frame, p1, p2):
    inicio = p1
    fim = p2

    x = 0
    y =  1

    #parte = frame

    parte = frame[inicio[y]:fim[y], inicio[x]:fim[x]]

    if parte.shape[0]>1 and parte.shape[1] > 1: 

        hsv = cv2.cvtColor(parte, cv2.COLOR_BGR2HSV)
        mini = np.array([135, 25,25])#, dtype=np.uint8)
        maxi = np.array([175, 255,255])#, dtype=np.uint8)
        mask = cv2.inRange(hsv, mini , maxi)

        cv2.imshow('Mascara', mask)

        cont = np.sum(mask)/255

        text_cv(parte, (1,45), '{}'.format(cont))
        cv2.imshow("Slice", parte)

        if cont > 1000: 
            return True



    return False

def faz_analise(frame, resultados): 

    magenta_dog = False
    p1_dog = (400,400)
    p2_dog = (401, 401)

    chair = False
    p1_chair = -1
    p2_chair = -1

    for r in resultados:
        if r[0]=="dog" or r[0]=='person':
            magenta = check_magenta(frame, r[2], r[3])

            if magenta:
                magenta_dog = True
                p1_dog = r[2]
                p2_dog = r[3]

        if r[0]=='chair': 
            chair = True
            p1_chair = r[2]
            p2_chair = r[3]
    x = 0
    y = 1

    acima = False 
    entre = False


    if magenta_dog: 
        xdog = int((p1_dog[x] + p2_dog[x])/2)

        if chair:
            if p2_dog[y] < p1_chair[y]:
                acima = True

            if p1_chair[x] <= xdog <= p2_chair[x]:
                entre = True

            if acima and entre: 
                text_cv(frame, (50, 400), 'Cao magenta acima da cadeira', 2, 2)
    return frame


if __name__ == "__main__":


    # Checando se os arquivos necessários existem
    check_exists_size(proto, 29353)
    check_exists_size(model, 23147564)
    check_exists_size(video, 1313000)

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)

    # cria a rede neural
    net = cv2.dnn.readNetFromCaffe(proto, model)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]   

    CONFIDENCE = 0.7
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        saida, resultados = detect(frame)

        print(type(resultados))
        print(resultados)



        saida_analise = faz_analise(frame, resultados)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('Resultado', saida_analise)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


