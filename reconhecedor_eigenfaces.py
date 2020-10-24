import cv2

camera = cv2.VideoCapture(r"C:\FFOutput\kristel5.mp4")

detector_facial = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classficadorEigen.yml")

largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    conectado, imagem = camera.read()

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.2, minSize=(100, 100))

    for x, y, l, a in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 0), 2)
        id, confianca = reconhecedor.predict(imagem_face)
        if id == 1:
            nome = 'Ines'
        elif id == 2:
            nome = 'Kristel'
        else:
            nome = 'Unknown'
        cv2.putText(imagem, str(nome), (x, y + (a + 30)), font, 2, (255, 255, 255))

    cv2.imshow('Reconhecimento Facial', imagem)
    if cv2.waitKey(20) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()