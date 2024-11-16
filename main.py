import cv2
import numpy as np
import time

# Função para calcular a distância em metros a partir do deslocamento em pixels
def pixel_to_meter(pixels, reference_length_pixels, reference_length_meters):
    """
    Converte deslocamento em pixels para metros.
    """
    return (pixels / reference_length_pixels) * reference_length_meters

# Configurações da referência
REFERENCE_LENGTH_PIXELS = 100  # Tamanho do QR code em pixels na imagem inicial
REFERENCE_LENGTH_METERS = 0.2  # Tamanho real do QR code em metros (exemplo: 20 cm)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)  # 0 para webcam, ou caminho para um vídeo

if not cap.isOpened():
    print("Erro ao acessar a câmera. Verifique a conexão.")
    exit()

# Inicializa o detector de QR code
qr_detector = cv2.QRCodeDetector()

# Posição inicial e variáveis de temporizador
initial_position = None
stable_position = None
stable_start_time = None
tracking_active = False

print("Pressione 'q' para sair.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo. Finalizando...")
        break

    # Detecta QR code na imagem
    data, bbox, _ = qr_detector.detectAndDecode(frame)

    if bbox is not None:
        # Converte a borda para inteiros
        bbox = np.int32(bbox)

        # Calcula o centro do QR code
        center_x = int(np.mean(bbox[:, 0, 0]))
        center_y = int(np.mean(bbox[:, 0, 1]))

        # Desenha o contorno do QR code e seu centro
        frame = cv2.polylines(frame, [bbox], True, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, "QR Code Detectado", (center_x, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Verifica se o QR code está estável
        if stable_position is None:
            stable_position = (center_x, center_y)
            stable_start_time = time.time()
        else:
            displacement_pixels = np.sqrt(
                (center_x - stable_position[0]) ** 2 +
                (center_y - stable_position[1]) ** 2
            )
            if displacement_pixels < 5:  # Considera estável se o movimento for menor que 5 pixels
                if time.time() - stable_start_time >= 3:  # Verifica se está estável por 3 segundos
                    if not tracking_active:
                        print("QR code estabilizado. Iniciando rastreamento...")
                        initial_position = (center_x, center_y)
                        tracking_active = True
                else:
                    cv2.putText(frame, "QR Code Estabilizando...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Reseta o temporizador se houver movimento significativo
                stable_position = (center_x, center_y)
                stable_start_time = time.time()
                tracking_active = False

        # Calcula o deslocamento se o rastreamento estiver ativo
        if tracking_active and initial_position is not None:
            displacement_pixels = np.sqrt(
                (center_x - initial_position[0]) ** 2 +
                (center_y - initial_position[1]) ** 2
            )
            displacement_meters = pixel_to_meter(
                displacement_pixels, REFERENCE_LENGTH_PIXELS, REFERENCE_LENGTH_METERS
            )

            # Exibe o deslocamento em tempo real na tela
            cv2.putText(frame, f"Deslocamento: {displacement_meters:.2f} m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Deslocamento: {displacement_pixels:.2f} pixels ({displacement_meters:.2f} metros)")
    else:
        # QR code não detectado
        cv2.putText(frame, "QR Code não detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Reseta estado se QR code desaparecer
        stable_position = None
        stable_start_time = None
        tracking_active = False

    # Exibe o vídeo em tempo real
    cv2.imshow("QR Code Tracker", frame)

    # Finaliza ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
