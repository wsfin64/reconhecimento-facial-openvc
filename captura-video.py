# Import the libraries
import cv2

# Reference your video file saved on your hard drive (mp4 format)
cap = cv2.VideoCapture(r"C:\FFOutput\kristel.mp4")
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")

amostra = 1
numero_amostra = 50
id = input('Digite o seu identificador: ')
largura, altura = 220, 220


while True:
    ret, frame = cap.read()

    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.2)

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = frame[y:y + a, x:x + l]
        regiao_cinza_olhos = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        olhos_detectados = classificadorOlho.detectMultiScale(regiao_cinza_olhos)

        for ox, oy, ol, oa in olhos_detectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                imagem_face = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))

                cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(amostra) + '.jpg', imagem_face)
                print('Foto ' + str(amostra) + ' capturada com sucesso')
                amostra += 1
    # Display the resulting image
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if amostra >= numero_amostra + 1:
        break


cap.release()
cv2.destroyAllWindows()
