import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lena.png 파일 읽기
image = cv2.imread('../data/Lena.png')

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel Filter 파라미터 설정
KSIZE = 3

# X 방향 미분 계산
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=KSIZE)

# Y 방향 미분 계산
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=KSIZE)

# Sobel 크기 계산 (magnitude)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# 0-255 범위로 정규화
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

print(f"Sobel X shape: {sobel_x.shape}, dtype: {sobel_x.dtype}")
print(f"Sobel Y shape: {sobel_y.shape}, dtype: {sobel_y.dtype}")
print(f"Sobel Magnitude shape: {sobel_magnitude.shape}, dtype: {sobel_magnitude.dtype}")

# 결과 표시 (matplotlib)
plt.figure(figsize=(15, 5))

plt.subplot(141)
plt.axis('off')
plt.title('Original Image')
plt.imshow(gray, cmap='gray')

plt.subplot(142)
plt.axis('off')
plt.title('Sobel X')
plt.imshow(np.abs(sobel_x), cmap='gray')

plt.subplot(143)
plt.axis('off')
plt.title('Sobel Y')
plt.imshow(np.abs(sobel_y), cmap='gray')

plt.subplot(144)
plt.axis('off')
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')

plt.tight_layout(pad=1.0)
plt.show()

# OpenCV 창에서 결과 표시
cv2.imshow('Original', gray)
cv2.imshow('Sobel X', np.uint8(np.abs(sobel_x)))
cv2.imshow('Sobel Y', np.uint8(np.abs(sobel_y)))
cv2.imshow('Sobel Magnitude', sobel_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

