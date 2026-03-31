import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일로 읽기)
image = cv2.imread('../data/BnW.png', 0)

# Otsu의 알고리즘을 통한 Thresholding
ret, image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("=" * 50)
print("(1) Otsu Thresholding 완료")
print("=" * 50)

# ===== Connected Component Labeling =====
num_labels, labels = cv2.connectedComponents(image_otsu)
print(f"\n연결된 Component 개수: {num_labels}")

# ===== Distance Transform 계산 =====
distance_transform = cv2.distanceTransform(image_otsu, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

print(f"Distance Transform 계산 완료")

# ===== Distance Transform 시각화 =====
fig1 = plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.axis('off')
plt.title('Otsu Thresholding Result')
plt.imshow(image_otsu, cmap='gray')

plt.subplot(132)
plt.axis('off')
plt.title('Connected Components Labels')
plt.imshow(labels, cmap='nipy_spectral')

plt.subplot(133)
plt.axis('off')
plt.title('Distance Transform')
plt.imshow(distance_transform_normalized, cmap='hot')

plt.tight_layout()
plt.show()

# ===== Interactive Connected Component Visualization =====
print("\n" + "=" * 50)
print("스페이스 키를 눌러 5개의 Random Component를 표시")
print("다른 키를 눌러 종료")
print("=" * 50 + "\n")

# 새 창 생성
cv2.namedWindow('Connected Components - Press SPACE for Random 5')

def show_random_components():
    """Random하게 5개의 component를 선택하여 표시"""
    # 0은 배경이므로 1부터 시작
    component_ids = np.random.choice(range(1, num_labels), min(5, num_labels - 1), replace=False)
    
    # 결과 이미지 생성 (컬러)
    result_image = np.zeros((image_otsu.shape[0], image_otsu.shape[1], 3), dtype=np.uint8)
    
    # Random 색상 생성 및 표시
    selected_colors = {}
    for comp_id in component_ids:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        selected_colors[comp_id] = color
        result_image[labels == comp_id] = color
    
    # 결과 표시
    cv2.imshow('Connected Components - Press SPACE for Random 5', result_image)
    
    print(f"표시된 Component IDs: {component_ids}")
    print(f"색상: {selected_colors}\n")

# 초기 화면 표시
show_random_components()

# 키보드 입력 처리
while True:
    key = cv2.waitKey(0)
    
    if key == ord(' '):  # 스페이스 키
        show_random_components()
    else:  # 다른 키 (ESC 또는 다른 키)
        break

cv2.destroyAllWindows()

print("프로그램 종료")

