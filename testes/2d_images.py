import cv2
import time
import queue
import threading
import numpy as np
import csv

# ------------------------------
# Configurações Gerais
# ------------------------------
CAM_INDEX = 0             # Índice da câmera (0, 1, etc.)
FRAME_WIDTH = 640         # Ajuste conforme necessidade
FRAME_HEIGHT = 480
FRAME_FPS = 60            # Tentar 60 FPS (verifique se a câmera suporta!)
SCALE_FACTOR = 1.0 / 500.0 # px -> metros (ajuste de acordo com seu setup)

# Para armazenar resultados de detecção: [(time, x_px, y_px), ...]
positions = []
# Para análise de frequência de detecção
detection_times = []
# Contadores globais
frame_count = 0
detection_count = 0

# Fila de frames
frame_queue = queue.Queue(maxsize=200)

# Flag global para parar threads (se preferir).
stop_threads = False

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
        # Enfileira o frame rapidamente
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # Se a fila estiver cheia, descartamos frames pra não travar
            # ou poderíamos usar .get() aqui pra liberar espaço, mas vamos descartar
            pass

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
                # Calcula centro do QR
                # pts[0] = array com 4 pontos: [[x0,y0], [x1,y1], ...]
                cx = int(np.mean(pts[0][:, 0]))
                cy = int(np.mean(pts[0][:, 1]))

                detection_count += 1
                detection_times.append(time.time())
                positions.append((time.time(), cx, cy))
        else:
            # Se não há frames na fila, dorme um pouco
            time.sleep(0.001)

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
    
    # 2) Cria e inicia as threads
    t_capture = threading.Thread(target=capture_thread_func, args=(cap,), daemon=True)
    t_decode = threading.Thread(target=decode_thread_func, daemon=True)

    t_capture.start()
    t_decode.start()

    # 3) Espera o usuário pressionar ENTER para encerrar
    input("Pressione ENTER para encerrar...\n")

    # Sinaliza para threads pararem
    stop_threads = True
    time.sleep(0.5)  # pequeno intervalo pra garantir que threads finalizem

    cap.release()
    print("Câmera liberada, threads encerradas.")
    
    # 4) Fazemos pós-processamento
    process_results()


def process_results():
    """Calcula velocidades, frequência e salva CSV."""
    if len(positions) < 2:
        print("Poucos dados de detecção para análise.")
        return

    # Converte para numpy
    pos_np = np.array(positions)  # shape (N, 3) -> [time, x_px, y_px]
    t = pos_np[:, 0]
    x_px = pos_np[:, 1]
    y_px = pos_np[:, 2]

    print(f"Total de leituras do QR: {len(positions)}")

    # --- Suavização (opcional) ---
    # Por exemplo, média móvel com janela 5:
    def moving_average(a, n=5):
        return np.convolve(a, np.ones(n), 'valid') / n

    xs = moving_average(x_px, 5)
    ys = moving_average(y_px, 5)
    ts = t[2:-2]  # ajusta pro tamanho de xs, ys

    if len(xs) < 2:
        print("Poucos pontos após suavização, sem cálculo de velocidade.")
        return
    
    # Converte px -> metros
    xs_m = xs * SCALE_FACTOR
    ys_m = ys * SCALE_FACTOR

    # Calcula velocidades
    dt = np.diff(ts)
    dx = np.diff(xs_m)
    dy = np.diff(ys_m)

    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)
    tv = ts[1:]

    # Exibe info no terminal
    print(f"Primeiro tempo: {t[0]:.3f} s, Último tempo: {t[-1]:.3f} s")
    print(f"Velocidade média: {np.mean(v):.3f} m/s")
    print(f"Velocidade máx: {np.max(v):.3f} m/s")

    # Frequência de detecção
    detection_times_np = np.array(detection_times)
    if len(detection_times_np) > 1:
        freq_dt = np.diff(detection_times_np)
        freq = 1.0 / freq_dt
        freq_media = np.mean(freq)
        print(f"Frequência média de detecção: {freq_media:.2f} Hz")
    else:
        print("Frequência média de detecção: insuficiente para cálculo.")

    # Salva CSV (opcional)
    with open('dados.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["time","X_px","Y_px","X_m","Y_m","Vx_ms","Vy_ms","V_ms"])
        for i in range(len(v)):
            writer.writerow([
                tv[i],
                xs[i+1], ys[i+1],
                xs_m[i+1], ys_m[i+1],
                vx[i], vy[i], v[i]
            ])
    print("Resultados salvos em dados.csv")


if __name__ == "__main__":
    main()
