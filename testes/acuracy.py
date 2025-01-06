import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import csv

# Configurações iniciais da câmera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

qr = cv2.QRCodeDetector()
is_recording = False
positions = []
scale_factor = 1.0 / 500.0

# Variáveis pra cálculo de frequência e acurácia
frame_count = 0
detection_count = 0
detection_times = []  # guardamos o tempo de cada detecção pra calcular frequência

# Função pra fazer média móvel (usada pra suavizar ruídos no X e Y se quiser)
def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), 'valid') / n

# Função pra desenhar linha tracejada entre pontos p1 e p2
def draw_dashed_line(img, p1, p2, color=(0, 0, 255), thickness=2, gap=10):
    """
    Desenha uma linha tracejada entre p1 e p2 no frame 'img'.
    gap define o tamanho do "pulo" entre cada segmento.
    """
    dist = np.hypot((p2[0] - p1[0]), (p2[1] - p1[1]))
    dash_count = int(dist // gap)
    if dash_count <= 0:
        return

    # Vetor unitário na direção p1->p2
    vx = (p2[0] - p1[0]) / dist
    vy = (p2[1] - p1[1]) / dist

    # Desenha segmentos "tracejados"
    for i in range(dash_count):
        start_pt = (int(p1[0] + vx * gap * i), int(p1[1] + vy * gap * i))
        end_pt = (int(p1[0] + vx * gap * (i + 0.5)),
                  int(p1[1] + vy * gap * (i + 0.5)))
        cv2.line(img, start_pt, end_pt, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # cada frame processado

    # Pré-processamento simples (Threshold) - pode testar pra ver se melhora a detecção
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Leitura do QR code
    data, pts, _ = qr.detectAndDecode(gray)

    h, w = frame.shape[:2]
    origin = (50, h - 50)

    # Desenha eixos no canto inferior esquerdo
    cv2.arrowedLine(frame, origin, (origin[0] + 40, origin[1]), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, 'X', (origin[0] + 45, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1] - 40), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, 'Y', (origin[0] - 15, origin[1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Se detectou QRCode
    if pts is not None and data:
        pts = pts[0]
        x = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4)
        y = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4)

        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        detection_count += 1
        detection_times.append(time.time())

        if is_recording:
            positions.append((time.time(), x, y))
            print(f"t={positions[-1][0]:.3f}, X_px={positions[-1][1]}, Y_px={positions[-1][2]}")

    # Desenha linha tracejada em tempo real ligando os últimos pontos
    if len(positions) > 1:
        p1 = positions[-2]
        p2 = positions[-1]
        # p1[1], p1[2] = X, Y
        draw_dashed_line(frame, (p1[1], p1[2]), (p2[1], p2[2]), color=(255, 0, 0), thickness=2, gap=10)

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

# Se gravamos pelo menos alguns pontos, geramos gráficos e salvamos CSV
if len(positions) > 5:
    pos = np.array(positions)
    t = pos[:, 0]
    x = pos[:, 1]
    y = pos[:, 2]

    # Suavização (janela de 5)
    xs = moving_average(x, 5)
    ys = moving_average(y, 5)
    ts = t[2:-2]  # ajusta o tamanho para bater com xs, ys

    # Converte px -> metros
    xs_m = xs * scale_factor
    ys_m = ys * scale_factor

    # Calcula velocidades
    dt = np.diff(ts)
    dx = np.diff(xs_m)
    dy = np.diff(ys_m)
    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)
    tv = ts[1:]

    # Plot da trajetória
    plt.figure()
    plt.title("Trajetória (X vs Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.plot(xs_m, ys_m, 'o-')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('trajetoria.png', dpi=150)
    plt.close()

    # Plot da velocidade
    plt.figure()
    plt.title("Velocidade ao longo do tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade (m/s)")
    plt.plot(tv, v, 'o-r')
    plt.grid(True)
    plt.savefig('velocidade.png', dpi=150)
    plt.close()

    # Plot da frequência de detecção (detections/s)
    detection_times = np.array(detection_times)
    if len(detection_times) > 1:
        freq_t = detection_times[1:]
        freq_dt = np.diff(detection_times)
        freq = 1.0 / freq_dt  # detections per second em cada intervalo

        plt.figure()
        plt.title("Frequência de Detecção (detections/s)")
        plt.xlabel("Tempo de Detecção (s)")
        plt.ylabel("Frequência (Hz)")
        plt.plot(freq_t, freq, '-o')
        plt.grid(True)
        plt.savefig('frequencia.png', dpi=150)
        plt.close()
    else:
        print("Poucas detecções para plotar frequência de detecção.")

    # Acurácia: quantos frames detectaram QR / total de frames
    accuracy = detection_count / float(frame_count) * 100.0
    print(f"Acurácia de Leitura: {accuracy:.2f}% (QR detectado em {detection_count} de {frame_count} frames)")

    # Salvando dados em CSV
    with open('dados.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["time", "X_m", "Y_m", "Vx_ms", "Vy_ms", "V_ms"])
        for i in range(len(v)):
            w.writerow([tv[i], xs_m[i+1], ys_m[i+1], vx[i], vy[i], v[i]])

    print("Dados salvos em dados.csv")
    print("Gráficos salvos: trajetoria.png, velocidade.png, frequencia.png")
else:
    print("Poucos dados registrados para gerar gráficos e CSV.")
