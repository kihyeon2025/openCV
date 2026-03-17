# 📸 Industrial Computer Vision (산업 컴퓨터 비전)

본 저장소는 산업 컴퓨터 비전 과목의 실습 내용을 정리한 프로젝트입니다.  
OpenCV를 활용하여 이미지 처리 및 영상 분석 기초를 학습합니다.

---

## 🛠️ 개발 환경

- Python 3.x
- OpenCV (cv2)
- NumPy

---

## 📅 Week 2 - 기본 영상 처리 및 사용자 인터페이스

### ✅ 주요 내용

### 1️⃣ 이미지 출력
- Lena 이미지를 컬러로 읽고 화면에 출력
- OpenCV `imshow()` 함수 사용

---

### 2️⃣ 마우스 이벤트 기반 도형 그리기
- 사각형 (Rectangle)
- 직선 (Line)
- 화살표 직선 (Arrowed Line)

👉 OpenCV Mouse Callback 사용

---

### 3️⃣ 키보드 입력 기반 모드 전환

| 키 | 기능 |
|----|------|
| r | 사각형 모드 |
| l | 직선 모드 |
| a | 화살표 모드 |

👉 키 입력으로 그리기 모드 변경

---

### 4️⃣ 이미지 저장

- 키보드 `w` 입력 시
- 현재 이미지 저장 → `lena_draw.png`

👉 OpenCV `imwrite()` 사용

---

### 5️⃣ 초기화 기능

- 키보드 `c` 입력 시
- 모든 도형 제거 후 원본 Lena 이미지 복원

---

### 6️⃣ 프로그램 종료

- `ESC` 키 입력 시 종료

---

## 📅 Week 3 - 영상 처리 기법

### ✅ 주요 내용

### 1️⃣ Grayscale 변환
- 컬러 이미지를 흑백으로 변환
- `cv2.cvtColor()` 사용

---

### 2️⃣ Histogram Equalization
- 흑백 영상 대비 향상
- `cv2.equalizeHist()` 사용

📌 히스토그램 평활화는 픽셀 분포를 균일하게 만들어  
이미지의 대비를 향상시키는 기법이다 :contentReference[oaicite:0]{index=0}  

---

### 3️⃣ Gamma Correction
- 밝기 조절 (비선형 변환)

---

### 4️⃣ HSV 색공간 변환
- BGR → HSV 변환
- 채널 분리 (H, S, V)

---

### 5️⃣ 필터 적용

| 채널 | 필터 |
|------|------|
| H | Median Filter |
| S | Gaussian Filter |
| V | Bilateral Filter |

👉 채널 특성에 맞는 필터 적용

---

## 📂 프로젝트 구조
