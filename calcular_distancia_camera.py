import cv2
import numpy as np
import math

# Se você já tiver calibrado a câmera (com QR ou tabuleiro), carregue a matriz:
# Mas se estiver usando apenas "fator de escala", ou calibração manual, 
# e *não* tiver camera_matrix/dist_coeffs confiáveis, você pode colocar algo aproximado
# ou usar np.eye(3) e np.zeros(5), mas vai dar erro grande.

# Exemplo: carregando de um "camera_qr.npz" (calibrado com VÁRIAS fotos do QR).
try:
    calib_data = np.load("camera_qr.npz")
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs   = calib_data["dist_coeffs"]
    print("Carreguei camera_matrix e dist_coeffs de camera_qr.npz")
except:
    print("Falha ao carregar camera_qr.npz. Usando matriz identidade (ruim).")
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs   = np.zeros(5, dtype=np.float32)

# ----------------------------------------------------
# 1) AJUSTE EXATO DO TAMANHO DO QR CODE
# Se seu QR Code REALMENTE tiver 11.7 cm (sem margens),
# ok. Mas se você mediu e descobriu que a parte quadrada
# do QR tem, por ex., 10.5 cm, ajuste aqui.
qr_size = 11.7  # cm (verifique de verdade)

# ----------------------------------------------------
# 2) DEFINIÇÃO DOS CANTOS 3D
# (0,0,0) -> top-left
# (qr_size, 0, 0) -> top-right
# (qr_size, qr_size, 0) -> bottom-right
# (0, qr_size, 0) -> bottom-left
object_points_3d = np.array([
    [0.0,      0.0,      0.0],
    [qr_size,  0.0,      0.0],
    [qr_size,  qr_size,  0.0],
    [0.0,      qr_size,  0.0]
], dtype=np.float32)

# ----------------------------------------------------
# 3) OFFSET DO CENTRO ÓPTICO
# Se você está medindo "na régua" da ponta da lente até o QR, mas o solvePnP
# mede do centro ótico (lá dentro da câmera), você pode subtrair uns 2 cm ou 5 cm
# (dependendo da câmera). Ajuste esse valor *após testes*.
lens_offset_cm = 0.0  # Se achar que deve tirar 5 cm, ponha lens_offset_cm=5.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Não abriu webcam.")
    exit()

qr_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data, points, _ = qr_detector.detectAndDecode(frame)
    
    if points is not None and len(points) > 0 and data != "":
        # 'points' tem shape (1, 4, 2). Vamos extrair e converter:
        corners_2d = points[0].astype(np.float32)
        
        # DICA DE DEBUG: desenhar cada canto de cor diferente, pra ver a ordem
        # corners_2d[0] -> top-left
        # corners_2d[1] -> top-right
        # corners_2d[2] -> bottom-right
        # corners_2d[3] -> bottom-left
        
        # Exemplo: desenha círculos coloridos para debug
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]  # BGR
        corner_names = ["TL", "TR", "BR", "BL"]
        for i, c in enumerate(corners_2d):
            cx, cy = int(c[0]), int(c[1])
            cv2.circle(frame, (cx, cy), 5, colors[i], -1)
            cv2.putText(frame, corner_names[i], (cx+5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # Se suspeitar que a ordem está invertida, reordene:
        # Exemplo se top-left e top-right estão trocados:
        # corners_2d = corners_2d[[1,0,3,2], :]

        # solvePnP
        retval, rvec, tvec = cv2.solvePnP(
            object_points_3d, 
            corners_2d, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if retval:
            # Distância do centro ótico da câmera até o QR
            dist = np.linalg.norm(tvec)
            
            # Aplica offset se você quiser compensar a medição "na régua"
            dist_real = dist - lens_offset_cm
            
            # Mostra no canto
            cv2.putText(frame, f"Dist sem offset: {dist:.2f} cm", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Dist c/ offset : {dist_real:.2f} cm", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # (Opcional) Desenha eixos
            # A "length" define o tamanho do eixo em cm (a ser desenhado na imagem)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 5.0)
        
        # Desenha contorno do QR (linhas verdes)
        pts = corners_2d.reshape((-1,1,2)).astype(int)
        cv2.polylines(frame, [pts], True, (0,255,0), 2)
        
        # Conteúdo do QR
        cv2.putText(frame, f"{data}", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    cv2.imshow("Medicao QR Code", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
