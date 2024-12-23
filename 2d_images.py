import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import csv

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

qr = cv2.QRCodeDetector()
is_recording = False
positions = []

scale_factor = 1.0 / 500.0  # px -> metros (ajuste se precisar)
frame_count = 0
detection_count = 0
detection_times = []

background = None
background_velocity = None
captured_background = False

def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), 'valid') / n

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data, pts, _ = qr.detectAndDecode(gray)

    # Desenha eixos (opcional)
    h, w = frame.shape[:2]
    origin = (50, h - 50)
    cv2.arrowedLine(frame, origin, (origin[0] + 40, origin[1]), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, 'X', (origin[0] + 45, origin[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1] - 40), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, 'Y', (origin[0] - 15, origin[1] - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Se detectou QR
    if pts is not None and data:
        pts = pts[0]
        x_px = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4)
        y_px = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4)

        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
        cv2.circle(frame, (x_px, y_px), 5, (0, 0, 255), -1)

        detection_count += 1
        detection_times.append(time.time())

        if is_recording:
            positions.append((time.time(), x_px, y_px))
            print(f"t={positions[-1][0]:.3f}, X_px={x_px}, Y_px={y_px}")

    # Instruções na tela
    cv2.putText(frame, "[r] Gravar/Parar   [c] Capturar Bg   [q] Sair",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        is_recording = not is_recording
        if is_recording:
            print("Iniciando gravação...")
        else:
            print("Parando gravação...")
    elif k == ord('c'):
        # Captura dois backgrounds diferentes: um pra trajetória, outro pra velocidade
        background = frame.copy()
        background_velocity = frame.copy()
        captured_background = True
        print("Background capturado!")
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Se não capturamos background, não gera as imagens
if not captured_background:
    print("Nenhum background capturado. Saindo sem gerar imagens.")
    exit()

if len(positions) <= 1:
    print("Poucos dados registrados para gerar gráficos.")
    exit()

# Arrays numpy
pos = np.array(positions)
t = pos[:, 0]
x = pos[:, 1]
y = pos[:, 2]

# Suaviza se quiser
xs = moving_average(x, 5)
ys = moving_average(y, 5)
ts = t[2:-2]
if len(xs) < 2:
    print("Poucos pontos após suavização, abortando.")
    exit()

xs_m = xs * scale_factor
ys_m = ys * scale_factor

dt = np.diff(ts)
dx = np.diff(xs_m)
dy = np.diff(ys_m)
vx = dx / dt
vy = dy / dt
v = np.sqrt(vx**2 + vy**2)
tv = ts[1:]

# ----------------------------------------------------------------------------
# 1) Desenha TRAJETÓRIA (pontos originais) em backgroundTrajectory
# ----------------------------------------------------------------------------
backgroundTrajectory = background.copy()
for i in range(len(x) - 1):
    p1 = (int(x[i]), int(y[i]))
    p2 = (int(x[i+1]), int(y[i+1]))
    cv2.line(backgroundTrajectory, p1, p2, (0, 255, 0), 2)

cv2.imwrite('trajetoria.png', backgroundTrajectory)
print("Trajetória salva em 'trajetoria.png'")

# ----------------------------------------------------------------------------
# 2) Desenha VELOCIDADE (pontos originais) em background_velocity
# ----------------------------------------------------------------------------
backgroundVel = background_velocity.copy()
num_segments = len(x) - 1
for i in range(num_segments):
    p1 = (int(x[i]), int(y[i]))
    p2 = (int(x[i+1]), int(y[i+1]))
    cv2.line(backgroundVel, p1, p2, (255, 255, 0), 2)

    # Escreve a velocidade no meio
    mpx = (x[i] + x[i+1]) / 2.0
    mpy = (y[i] + y[i+1]) / 2.0
    if i < len(v):
        cv2.putText(backgroundVel, f"{v[i]:.2f} m/s",
                    (int(mpx), int(mpy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite('velocidade.png', backgroundVel)
print("Velocidade salva em 'velocidade.png'")

# ----------------------------------------------------------------------------
# 3) Calcula frequência e acurácia
# ----------------------------------------------------------------------------
detection_times = np.array(detection_times)
if len(detection_times) > 1:
    freq_dt = np.diff(detection_times)
    freq_vals = 1.0 / freq_dt
    freq_media = np.mean(freq_vals)
else:
    freq_media = 0.0

accuracy = detection_count / float(frame_count) * 100.0
print(f"Frequência média de detecção: {freq_media:.2f} Hz")
print(f"Acurácia: {accuracy:.2f}% (QR detectado em {detection_count} de {frame_count} frames)")

# ----------------------------------------------------------------------------
# 4) Salva CSV (se quiser)
# ----------------------------------------------------------------------------
with open('dados.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["time","X_px","Y_px","X_m","Y_m","Vx_ms","Vy_ms","V_ms"])
    for i in range(len(v)):
        w.writerow([
            tv[i],
            x[i+1], y[i+1],
            xs_m[i+1], ys_m[i+1],
            vx[i], vy[i], v[i]
        ])

print("Dados salvos em 'dados.csv'")
