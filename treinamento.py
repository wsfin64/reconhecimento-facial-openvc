import cv2
import os
import numpy

eigenface = cv2.face.EigenFaceRecognizer_create()

fisherface = cv2.face.FisherFaceRecognizer_create()

lbph = cv2.face.LBPHFaceRecognizer_create()


def get_imagem_com_id():

    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]

    faces = []

    ids = []

    for caminho_imagem in caminhos:
        image_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Face', image_face)
        # cv2.waitKey(10)

        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        #print(id)

        ids.append(id)
        faces.append(image_face)
    return numpy.array(ids), faces  # retorna os ids e imagens como arrays que serão usando no reconhecimento


ids, faces = get_imagem_com_id()
# print(faces)

print('Treinando....')

eigenface.train(faces, ids)  # Aprendizagem supervisionada
eigenface.write('classficadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')
