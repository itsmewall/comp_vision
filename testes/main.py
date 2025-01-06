import cv2
import cvzone
from pyzbar import pyzbar as bar
import time
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

positions = []

print("press 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de vídeo.")
        break

    result = bar.decode(frame)
    
    for data in result:
        x, y, w, h = data.rect
        x_center = x + w/2
        y_center = y + h/2
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.circle(frame, (int(x_center), int(y_center)), 5, (0,0,255), -1)

        timestamp = time.time()
        positions.append((timestamp, x_center, y_center))

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if positions:
    times = [p[0] for p in positions]
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    plt.figure(figsize=(8,6))
    plt.title("trajetória do QR Code")
    plt.xlabel("Posição X (pixels)")
    plt.ylabel("Posição Y (pixels)")
    plt.plot(xs, ys, marker='o')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.savefig('trajetoria_qr.png', dpi=150)
    plt.show()
else:
    print("Nenhum dado de posição foi registrado")