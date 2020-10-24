import cv2

image_path = 'Capturar.JPG'

cascade_path = r'C:\Users\Wellington\PycharmProjects\facial-recognition\haarcascade_frontalface_default.xml'

carrega_algoritmo = cv2.CascadeClassifier(cascade_path)  # Carregando o algoritomo

img = cv2.imread(image_path) # Carregando a imagem

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertendo a imagem para preto e branco

faces = carrega_algoritmo.detectMultiScale(gray, 1.3, 10, minSize=(100, 100))

print(faces)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
