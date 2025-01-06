import cv2
import numpy as np

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Câmera não pode ser acessada.")
    exit()

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Inicializa o detector de QR Code
qr_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        break
    
    # Detecta o QR Code e decodifica o conteúdo
    data, points, _ = qr_detector.detectAndDecode(frame)

    # Se houver um QR Code detectado
    if points is not None and len(points) > 0:
        # 'points' possui as coordenadas dos 4 cantos do QR Code
        pts = np.int32(points).reshape((-1, 1, 2))

        # Desenha o contorno do QR Code em verde
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Calcula o centro do QR Code
        qr_center_x = int(np.mean(pts[:, 0, 0]))
        qr_center_y = int(np.mean(pts[:, 0, 1]))

        # Desenha uma bolinha vermelha no centro do QR Code
        cv2.circle(frame, (qr_center_x, qr_center_y), 5, (0, 0, 255), -1)

        # Exibe o conteúdo decodificado do QR Code na tela
        cv2.putText(frame, f"QR Data: {data}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Exibe o vídeo com a detecção do QR Code
    cv2.imshow("Deteccao QR Code", frame)

    # Pressione ESC para sair do loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
