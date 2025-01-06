import cv2
import numpy as np
import csv
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Câmera não pode ser acessada.")
    exit()

# Ajusta a resolução (pode alterar se quiser maior ou menor)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Inicializa o detector de QR Code
qr_detector = cv2.QRCodeDetector()

# Lista para guardar a trajetória (x, y) do QR
trajectory_points = []

# Se quiser salvar em CSV, abra um arquivo pra gravar
# 'newline=""' evita linhas em branco no Windows
csv_file = open('trajetoria_qr.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "PosX", "PosY"])  # Cabeçalho

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        break

    # Converte para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta e decodifica o QR Code na imagem
    data, points, _ = qr_detector.detectAndDecode(gray)

    # Se houver um QR Code detectado e data não for vazio
    if points is not None and len(points) > 0 and data != "":
        pts = np.int32(points).reshape((-1, 1, 2))

        # Desenha o contorno do QR Code em verde
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Calcula o centro do QR Code
        qr_center_x = int(np.mean(pts[:, 0, 0]))
        qr_center_y = int(np.mean(pts[:, 0, 1]))

        # Desenha uma bolinha vermelha no centro do QR Code
        cv2.circle(frame, (qr_center_x, qr_center_y), 5, (0, 0, 255), -1)

        # Adiciona o ponto atual à lista de trajetória
        trajectory_points.append((qr_center_x, qr_center_y))

        # Salva a posição no CSV com data/hora
        csv_writer.writerow([time.time(), qr_center_x, qr_center_y])

        # Exibe o conteúdo do QR Code
        cv2.putText(frame, f"QR Data: {data}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Desenha a linha da trajetória, ligando cada ponto consecutivo
    for i in range(1, len(trajectory_points)):
        cv2.line(frame,
                 trajectory_points[i - 1],
                 trajectory_points[i],
                 (255, 0, 255),
                 2)

    cv2.imshow("Deteccao e Trajetoria do QR Code", frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Fecha o arquivo CSV
csv_file.close()