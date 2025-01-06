import cv2
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Câmera não pode ser acessada.")
    exit()

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

qr_detector = cv2.QRCodeDetector()

trajectory_points = []

csv_file = open('trajetoria_qr.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "PosX", "PosY"])

output_dir = "imagens_salvas"
os.makedirs(output_dir, exist_ok=True)

snapshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    data, points, _ = qr_detector.detectAndDecode(gray)

    if points is not None and len(points) > 0 and data != "":
        pts = np.int32(points).reshape((-1, 1, 2))

        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        qr_center_x = int(np.mean(pts[:, 0, 0]))
        qr_center_y = int(np.mean(pts[:, 0, 1]))

        cv2.circle(frame, (qr_center_x, qr_center_y), 5, (0, 0, 255), -1)

        trajectory_points.append((qr_center_x, qr_center_y))

        csv_writer.writerow([time.time(), qr_center_x, qr_center_y])

        cv2.putText(frame, f"QR Data: {data}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for i in range(1, len(trajectory_points)):
        cv2.line(frame,
                 trajectory_points[i - 1],
                 trajectory_points[i],
                 (255, 0, 255),
                 2)

    cv2.imshow("Deteccao e Trajetoria do QR Code", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s') or key == ord('S'):
        snapshot_count += 1
        image_filename = os.path.join(output_dir, f"snapshot_{snapshot_count}.png")
        cv2.imwrite(image_filename, frame)
        
        if len(trajectory_points) > 1:
            xs = [p[0] for p in trajectory_points]
            # Inverte o Y antes de plotar
            ys = [frame_height - p[1] for p in trajectory_points]
            
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker='o', color='magenta', label='Trajetória QR')
            plt.title("Trajetória do QR Code (Eixo Y invertido manualmente)")
            plt.xlabel("Posição X (px)")
            plt.ylabel("Posição Y (px)")
            plt.grid(True)
            plt.legend()

            graph_filename = os.path.join(output_dir, f"trajetoria_{snapshot_count}.png")
            plt.savefig(graph_filename)
            plt.close()

            print(f"Imagem salva em: {image_filename}")
            print(f"Gráfico salvo em: {graph_filename}")
        else:
            print("Não há pontos suficientes para gerar um gráfico.")

cap.release()
cv2.destroyAllWindows()
csv_file.close()
