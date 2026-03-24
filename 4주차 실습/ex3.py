import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lena.png 파일 읽기
image = cv2.imread('../data/Lena.png')

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gabor Filter 파라미터 설정
KSIZE = 21
SIGMA = 3.0
LAMBD = 10.0  # 파장 (wavelength)
GAMMA = 0.5   # 종횡비 (aspect ratio)
PSI = 0       # 위상 오프셋

# 다양한 방향으로 Gabor Filter 적용
angles = [0, 45, 90, 135]  # 0도, 45도, 90도, 135도
filtered_images = []

print(f"Gabor Filter 적용 - KSIZE: {KSIZE}, SIGMA: {SIGMA}, LAMBD: {LAMBD}, GAMMA: {GAMMA}")

for angle in angles:
    # 각도를 라디안으로 변환
    theta = np.pi * angle / 180.0
    
    # Gabor 커널 생성
    kernel = cv2.getGaborKernel((KSIZE, KSIZE), SIGMA, theta, LAMBD, GAMMA, PSI)
    
    # 커널 정규화
    kernel = kernel / np.sum(np.abs(kernel))
    
    # Gabor Filter 적용
    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
    
    # 0-255 범위로 정규화
    filtered = np.uint8(255 * (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-6))
    
    filtered_images.append(filtered)
    print(f"Angle: {angle}° - Filter applied")

# 결과 표시 (matplotlib)
plt.figure(figsize=(14, 6))

plt.subplot(2, 3, 1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(gray, cmap='gray')

for i, (angle, filtered) in enumerate(zip(angles, filtered_images)):
    plt.subplot(2, 3, i + 2)
    plt.axis('off')
    plt.title(f'Gabor {angle}°')
    plt.imshow(filtered, cmap='gray')

plt.tight_layout(True)
plt.show()

# OpenCV 창에서 결과 표시
'''cv2.imshow('Original', gray)
for angle, filtered in zip(angles, filtered_images):
    cv2.imshow(f'Gabor Filter {angle}°', filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

