    import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일로 읽기)
image = cv2.imread('../data/BnW.png', 0)

# Otsu의 알고리즘을 통한 Thresholding
ret, image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 결과 시각화
plt.figure(figsize=(10, 4))

# 원본 이미지
plt.subplot(121)
plt.axis('off')
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Otsu Thresholding 결과
plt.subplot(122)
plt.axis('off')
plt.title(f'Otsu Thresholding (threshold={ret})')
plt.imshow(image_otsu, cmap='gray')

plt.tight_layout()
plt.show()

