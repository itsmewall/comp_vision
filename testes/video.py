import cv2
import time
import queue
import threading
import numpy as np
import csv

# ------------------------------
# Configurações Gerais
# ------------------------------
CAM_INDEX = 0              # Índice da câmera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 60             # Tentar 60 FPS (verifique se a câmera suporta)
SCALE_FACTOR = 1.0 / 500.0 # px -> metros (ajuste conforme seu setup)

# Fila de frames para decodificação
frame_queue = queue.Queue(maxsize=200)
# Lista para armazenar todos os frames (para slow motion depois)
all_frames = []

# Flag global para parar threads
stop_threads = False

# Para análise do QR
positions = []       # [(time, x_px, y_px)]
detection_times = []
frame_count = 0
detection_count = 0

# ----------------------------------
# Thread 1: Captura
# ----------------------------------
def capture_thread_func(cap):
    global frame_count, stop_threads

    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame. Encerrando thread de captura.")
            break

        frame_count += 1

        # Enfileira o frame rapidamente pra decodificação
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # descarta se fila cheia

        # Armazena o frame na lista para slow motion depois
        all_frames.append(frame)

    print("Thread de captura finalizada.")
    cap.release()

# ----------------------------------
# Thread 2: Decodificação do QR
# ----------------------------------
def decode_thread_func():
    global detection_count, stop_threads

    qr_detector = cv2.QRCodeDetector()

    while not stop_threads:
        if not frame_queue.empty():
            frame = frame_queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, pts, _ = qr_detector.detectAndDecode(gray)

            if pts is not None and data:
                cx = int(np.mean(pts[0][:, 0]))
                cy = int(np.mean(pts[0][:, 1]))

                detection_count += 1
                current_time = time.time()
                detection_times.append(current_time)
                positions.append((current_time, cx, cy))
        else:
            time.sleep(0.001)

    print("Thread de decodificação finalizada.")

# ----------------------------------
# Função principal
# ----------------------------------
def main():
    global stop_threads

    # 1) Inicializa a câmera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    if not cap.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    print("Iniciando threads de captura e decodificação...")
    t_capture = threading.Thread(target=capture_thread_func, args=(cap,), daemon=True)
    t_decode = threading.Thread(target=decode_thread_func, daemon=True)

    t_capture.start()
    t_decode.start()

    # 2) Exibição em tempo real
    #    Enquanto as threads rodam, mostramos o frame "ao vivo".
    #    (Lemos novamente da câmera só pra exibir, ou podemos exibir da fila,
    #     mas aqui mostro uma lógica simples lendo diretamente do cap)
    while True:
        ret, display_frame = cap.read()
        if not ret:
            # Provavelmente a câmera acabou ou deu erro
            break

        cv2.imshow("Ao Vivo", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Sai do loop, paramos a gravação
            break
        elif key == ord('s'):
            # Entra no slow motion playback com os frames armazenados
            slow_motion_playback(all_frames)
            # Depois que sair do slow motion, volta ao loop de exibição normal

    # 3) Finaliza
    stop_threads = True
    time.sleep(0.5)

    cv2.destroyAllWindows()
    print("Câmera encerrada.")

    # 4) Pós-processamento
    process_results()

def slow_motion_playback(frames, delay_ms=100):
    """
    Reproduz a lista de frames em slow motion,
    usando 'delay_ms' como tempo de exibição (100ms ~ 10FPS).
    """
    print("Entrando em Slow Motion Playback...")
    for i, f in enumerate(frames):
        cv2.imshow("Slow Motion", f)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow("Slow Motion")
    print("Saindo do Slow Motion Playback...")

# ----------------------------------
# Pós-processamento: velocidades e CSV
# ----------------------------------
def process_results():
    if len(positions) < 2:
        print("Poucos dados de detecção para análise.")
        return

    pos_np = np.array(positions)  # [time, x_px, y_px]
    t = pos_np[:, 0]
    x_px = pos_np[:, 1]
    y_px = pos_np[:, 2]

    print(f"Total de leituras do QR: {len(positions)}")

    def moving_average(a, n=5):
        return np.convolve(a, np.ones(n), 'valid') / n

    xs = moving_average(x_px, 5)
    ys = moving_average(y_px, 5)
    ts = t[2:-2]

    if len(xs) < 2:
        print("Poucos pontos após suavização, sem cálculo de velocidade.")
        return

    xs_m = xs * SCALE_FACTOR
    ys_m = ys * SCALE_FACTOR

    dt = np.diff(ts)
    dx = np.diff(xs_m)
    dy = np.diff(ys_m)
    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)
    tv = ts[1:]

    print(f"Primeiro tempo: {t[0]:.3f} s, Último tempo: {t[-1]:.3f} s")
    print(f"Velocidade média: {np.mean(v):.3f} m/s")
    print(f"Velocidade máx: {np.max(v):.3f} m/s")

    # Frequência de detecção
    det_times_np = np.array(detection_times)
    if len(det_times_np) > 1:
        freq_dt = np.diff(det_times_np)
        freq = 1.0 / freq_dt
        freq_media = np.mean(freq)
        print(f"Frequência média de detecção: {freq_media:.2f} Hz")
    else:
        print("Frequência média de detecção: insuficiente para cálculo.")

    # Salva CSV
    with open('dados.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["time","X_px","Y_px","X_m","Y_m","Vx_ms","Vy_ms","V_ms"])
        for i in range(len(v)):
            w.writerow([tv[i],
                        xs[i+1], ys[i+1],
                        xs_m[i+1], ys_m[i+1],
                        vx[i], vy[i], v[i]])


if __name__ == "__main__":
    main()
