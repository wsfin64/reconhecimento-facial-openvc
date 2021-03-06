import cv2
import numpy

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)

amostra = 1
numeroAmostras = 25
id = input('Digite o seu identificador: ')
largura, algura = 220, 220
print('Capturando as faces')


while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = imagem[y:y + a, x:x + l]
        regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        olhos_detectados = classificadorOlho.detectMultiScale(regiao_cinza_olho)

        for ox, oy, ol, oa in olhos_detectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if numpy.average(imagemCinza) > 110:  # luminosidade

                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, algura))
                    cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(amostra) + '.jpg', imagemFace)
                    print('Foto ' + str(amostra) + 'capturada com sucesso')
                    amostra += 1

    cv2.imshow('Face', imagem)
    cv2.waitKey(1)
    if amostra >= numeroAmostras + 1:
        break

print('faces capturadas com sucesso!')
camera.release()
cv2.destroyAllWindows()
