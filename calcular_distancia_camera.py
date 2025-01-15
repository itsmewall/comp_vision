import cv2
import numpy as np
import math

# Carregar os dados da calibração
try:
    calib_data = np.load("camera_qr.npz")
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
    print("[OK] Carreguei camera_matrix e dist_coeffs de 'camera_qr.npz'")
except:
    print("[ERRO] Falha ao carregar 'camera_qr.npz'. Usando matriz identidade (ruim).")
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

# Parâmetros do QR Code
qr_size = 11.5  # Tamanho real do QR Code (lado do quadrado em cm)
object_points_3d = np.array([
    [0.0,      0.0,      0.0],   # Top-Left
    [qr_size,  0.0,      0.0],   # Top-Right
    [qr_size,  qr_size,  0.0],   # Bottom-Right
    [0.0,      qr_size,  0.0]    # Bottom-Left
], dtype=np.float32)

# Ajuste de offset do centro óptico
lens_offset_cm = 5.0  # Ajuste para compensar a diferença física da lente

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERRO] Não abriu a webcam.")
    exit()

qr_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível capturar o frame.")
        break

    data, points, _ = qr_detector.detectAndDecode(frame)

    if points is not None and len(points) > 0 and data != "":
        corners_2d = points[0].astype(np.float32)

        # Desenhar pontos detectados
        for i, point in enumerate(corners_2d):
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"{i}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        retval, rvec, tvec = cv2.solvePnP(
            object_points_3d, 
            corners_2d, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if retval:
            dist = np.linalg.norm(tvec)
            dist_real = dist - lens_offset_cm

            # Exibir distância
            cv2.putText(frame, f"Dist sem offset: {dist:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Dist com offset: {dist_real:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Desenhar eixos para debug visual
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 10.0)

        # Desenhar contorno do QR Code
        pts = corners_2d.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "QR Code nao detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Medicao QR Code", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
