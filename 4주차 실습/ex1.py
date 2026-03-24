import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lena.png 파일 읽기
image = cv2.imread('../data/Lena.png')

# Unsharp Mask 파라미터 설정
KSIZE = 11
ALPHA = 2

# 가우시안 커널 생성
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

print(f"Kernel shape: {kernel.shape}, dtype: {kernel.dtype}, sum: {kernel.sum()}")

# Unsharp Mask 필터 적용
filtered = cv2.filter2D(image, -1, kernel)

# 결과 표시 (matplotlib)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.axis('off')
plt.title('Original Image')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis('off')
plt.title('Unsharp Mask Applied')
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout()
plt.show()

# OpenCV 창에서 결과 표시
cv2.imshow('Before - Original', image)
cv2.imshow('After - Unsharp Mask', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

