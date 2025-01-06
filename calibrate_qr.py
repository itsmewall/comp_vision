import cv2
import numpy as np
import glob

# Dimensão real do QR em cm
qr_size = 11.7

# Definição dos pontos 3D do QR (cantos) no mundo (Z=0)
# Ordem: top-left, top-right, bottom-right, bottom-left (padrão do QRCodeDetector)
objp = np.array([
    [0.0,     0.0,     0.0],
    [qr_size, 0.0,     0.0],
    [qr_size, qr_size, 0.0],
    [0.0,     qr_size, 0.0]
], dtype=np.float32)

# Listas para armazenar as correspondências 3D (objpoints) e 2D (imgpoints)
objpoints = []  # pontos 3D
imgpoints = []  # pontos 2D

# Inicializa o detector de QR Code
qr_detector = cv2.QRCodeDetector()

# Carrega as imagens da pasta (mude o path se quiser)
images = glob.glob('imgs/*.jpeg')

if not images:
    print("Nenhuma imagem encontrada na pasta 'qr_calib'!")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Erro ao carregar imagem {fname}")
        continue

    data, corners, _ = qr_detector.detectAndDecode(img)

    if corners is not None and len(corners) > 0 and data != "":
        # corners tem shape (1,4,2), vamos converter em float32
        corners_2d = corners[0].astype(np.float32)

        # corners_2d[0] = top-left, corners_2d[1] = top-right,
        # corners_2d[2] = bottom-right, corners_2d[3] = bottom-left
        # objp segue essa mesma ordem.

        # Guarda os 4 pontos 3D e 4 pontos 2D
        objpoints.append(objp)
        imgpoints.append(corners_2d)

        print(f"Imagem {fname} => QR detectado (data = {data})")
    else:
        print(f"Imagem {fname} => QR NÃO detectado ou data vazia.")

# Precisamos de pelo menos algumas imagens detectadas
if len(objpoints) < 3:
    print("Poucas imagens com QR detectado. Não é possível calibrar!")
    exit()

# Agora vamos calibrar
# Pegamos o tamanho do primeiro frame só pra referência (poderia ser mais robusto)
h, w = img.shape[:2]
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,    # lista de arrays (N x 4 x 3)
    imgpoints,    # lista de arrays (N x 4 x 2)
    (w, h),       # dimensão da imagem
    None, 
    None
)

print("Calibração concluída com QR Code!")
print("Ret =", ret)
print("Camera Matrix:\n", camera_matrix)
print("Dist Coeffs:\n", dist_coeffs)

# Salva em arquivo
np.savez("camera_qr.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

print("\nParâmetros salvos em 'camera_qr.npz'")
