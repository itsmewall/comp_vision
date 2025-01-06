import cv2
import numpy as np
import glob

# Parâmetros do tabuleiro
chessboard_size = (9, 6)  # número de cantos internos (linhas, colunas)
square_size = 2.5         # tamanho do quadradinho em cm (exemplo)

# Critério para sub-pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Listas para armazenar pontos 3D e 2D de todas as imagens
objpoints = []  # pontos 3D no mundo real
imgpoints = []  # pontos 2D no plano da imagem

# Cria o array de pontos 3D do tabuleiro (0,0,0), (1,0,0), ...
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1,2)
objp = objp * square_size  # escala para cm

# Carrega todas as imagens de um diretório, por exemplo
images = glob.glob('imgs/*.jpeg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Acha os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # Refina a posição dos cantos
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_subpix)

# Finalmente, calibra
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Calibração ok?", ret)
print("Camera Matrix:\n", camera_matrix)
print("Dist Coeffs:\n", dist_coeffs)

# Salva em um arquivo
np.savez("camera_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
