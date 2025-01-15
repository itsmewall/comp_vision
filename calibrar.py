import cv2
import numpy as np
import glob

# Parâmetros do tabuleiro
chessboard_size = (7, 7)  # Número de cantos internos (linhas, colunas)
square_size = 2.1         # Tamanho de cada quadradinho (em cm)

# Critério para refinamento dos cantos
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Listas para armazenar pontos 3D e 2D
objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Criação do padrão de pontos 3D
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
objp *= square_size  # Escala para cm

# Carregar imagens da pasta
images = glob.glob('imgs/*.jpg')  # Ajuste para seu diretório e extensão
total_images = len(images)
if total_images == 0:
    print("Nenhuma imagem encontrada. Verifique o caminho e a extensão.")
    exit()

# Processar as imagens
print(f"Total de imagens encontradas: {total_images}")
detected_images = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Erro ao carregar a imagem: {fname}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        detected_images += 1
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"[OK] Tabuleiro detectado na imagem: {fname}")
    else:
        print(f"[ERRO] Falha ao detectar o tabuleiro na imagem: {fname}")

# Exibir índice de aproveitamento
indice_aproveitamento = (detected_images / total_images) * 100
print(f"\nÍndice de aproveitamento: {indice_aproveitamento:.2f}%")

# Calibração da câmera
if detected_images < 3:
    print("Imagens insuficientes para calibração. Capture mais imagens.")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Resultados da calibração
print(f"\nRMS (re-projection error): {ret:.4f}")
print("Camera Matrix:\n", camera_matrix)
print("Dist Coefficients:\n", dist_coeffs)

# Salvar os dados
np.savez("camera_qr.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\nDados de calibração salvos em 'camera_qr.npz'")
