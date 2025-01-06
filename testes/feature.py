import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import csv

# Instruções:
# Pressione 'r' para iniciar/parar a gravação dos dados de posição do QR Code.
# Pressione 'q' para sair do programa.
#
# O script detecta um QR Code, registra sua posição ao longo do tempo,
# gera gráficos da trajetória (X vs Y) e da velocidade ao longo do tempo,
# além de salvar os dados em um arquivo CSV.
#
# Também desenha um pequeno eixo X,Y no canto inferior esquerdo da tela,
# e imprime no terminal todos os pontos registrados.

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

qr = cv2.QRCodeDetector()
is_recording = False
positions = []
scale_factor = 1.0/500.0

def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), 'valid') / n

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data, pts, _ = qr.detectAndDecode(gray)

    h, w = frame.shape[:2]
    origin = (50, h - 50)
    cv2.arrowedLine(frame, origin, (origin[0]+40, origin[1]), (0,0,255), 2, tipLength=0.3)
    cv2.putText(frame, 'X', (origin[0]+45, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1]-40), (0,255,0), 2, tipLength=0.3)
    cv2.putText(frame, 'Y', (origin[0]-15, origin[1]-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    if pts is not None and data:
        pts = pts[0]
        x = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4)
        y = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4)
        cv2.polylines(frame, [pts.astype(int)], True, (0,255,0), 2)
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        
        if is_recording:
            positions.append((time.time(), x, y))
            print(f"t={positions[-1][0]:.3f}, X_px={positions[-1][1]}, Y_px={positions[-1][2]}")

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        is_recording = not is_recording
        if is_recording:
            print("Iniciando gravação...")
        else:
            print("Parando gravação...")
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(positions) > 5:
    pos = np.array(positions)
    t = pos[:,0]
    x = pos[:,1]
    y = pos[:,2]
    xs = moving_average(x, 5)
    ys = moving_average(y, 5)
    ts = t[2:-2]
    xs_m = xs * scale_factor
    ys_m = ys * scale_factor
    dt = np.diff(ts)
    dx = np.diff(xs_m)
    dy = np.diff(ys_m)
    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)
    tv = ts[1:]

    plt.figure()
    plt.title("Trajetória (X vs Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.plot(xs_m, ys_m, 'o-')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('trajetoria.png', dpi=150)

    plt.figure()
    plt.title("Velocidade ao longo do tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade (m/s)")
    plt.plot(tv, v, 'o-r')
    plt.grid(True)
    plt.savefig('velocidade.png', dpi=150)

    with open('dados.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["time","X_m","Y_m","Vx_ms","Vy_ms","V_ms"])
        for i in range(len(v)):
            w.writerow([tv[i], xs_m[i+1], ys_m[i+1], vx[i], vy[i], v[i]])

    print("Dados salvos em dados.csv")
    print("Gráficos salvos: trajetoria.png e velocidade.png")
else:
    print("Poucos dados registrados.")
