import numpy as np
from numpy.linalg import svd
import cv2
import matplotlib.pyplot as plt

def tensor_train(A, r1, r2):
    N1, N2, N3 = A.shape
    A1 = A.reshape(N1, N2 * N3)
    U, s, V = np.linalg.svd(A1)
    G1 = U[:, :r1]  # N1 x r1
    V1 = np.diag(s[:r1]) @ V[:r1, :]  # r1 x (N2 * N3)
    V1 = V1.reshape(r1 * N2, N3)
    
    # Второй SVD
    U, s, V = np.linalg.svd(V1, full_matrices=False)
    G2 = U[:, :r2]  # r1 * N2 x r2
    print(f"Shape of G2 before reshape: {G2.shape}")
    G2 = G2.reshape(r1, N2, r2)  # r1 x N2 x r2
    G3 = np.diag(s[:r2]) @ V[:r2, :]  # r2 x N3
    
    return G1, G2, G3

img_color = cv2.imread('D:\\Users\\user\\Desktop\\lion.jpeg')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

def normalize_image(image):

    min_pixel = np.min(image)
    max_pixel = np.max(image)
    normalized_image = 255 * (image - min_pixel) / (max_pixel - min_pixel)

    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def constr(img,r1,r2):
    g1, g2, g3 = tensor_train(img, r1, r2)
    ans1 = np.tensordot(g1,g2,axes=[1,0])
    ans2 = np.tensordot(ans1,g3,axes=[2,0])
    return ans2

plt.figure(figsize=(12, 6))
plt.imshow(normalize_image(constr(img_color,100,3)))
plt.axis('off')
plt.tight_layout()
plt.show()

image = cv2.cvtColor(normalize_image(constr(img_color,100,3)), cv2.COLOR_BGR2RGB)
cv2.imwrite('ma.jpeg',image,[cv2.IMWRITE_JPEG_QUALITY, 90])