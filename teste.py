import cv2

cascade_path = 'haarcascade_frontalface_default.xml'

img_path = '17.jpg'

algoritmo = cv2.CascadeClassifier(cascade_path)

imagem = cv2.imread(img_path)

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = algoritmo.detectMultiScale(cinza, scaleFactor=1.3, minSize=(30, 30))

for (x, y, largura, altura) in faces:
    imagem = cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (255, 255, 0), 2)

cv2.imshow('Face', imagem)
cv2.waitKey()
cv2.destroyAllWindows()
