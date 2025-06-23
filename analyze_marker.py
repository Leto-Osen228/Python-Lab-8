import cv2
import numpy as np

# Загружаем изображение метки
marker = cv2.imread('Python-Lab-8/ref-point.jpg')
if marker is None:
    print("Не удалось загрузить ref-point.jpg")
    exit()

marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)

print(f"Размер маркера: {marker.shape}")
print(f"Средняя яркость: {np.mean(marker_gray):.2f}")
print(f"Мин яркость: {np.min(marker_gray)}")
print(f"Макс яркость: {np.max(marker_gray)}")

# Анализ краев
edges = cv2.Canny(marker_gray, 50, 150)

# Поиск контуров
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours)}")

# Анализ кругов
circles = cv2.HoughCircles(marker_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                          param1=50, param2=30, minRadius=10, maxRadius=0)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    print(f"Найдено кругов: {len(circles)}")
    for (x, y, r) in circles:
        print(f"Круг: центр ({x}, {y}), радиус {r}")

# Показываем результаты анализа
cv2.imshow('Original Marker', marker)
cv2.imshow('Marker Gray', marker_gray)
cv2.imshow('Edges', edges)

# Рисуем контуры и круги
result = marker.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

if circles is not None:
    for (x, y, r) in circles:
        cv2.circle(result, (x, y), r, (255, 0, 0), 2)
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

cv2.imshow('Analysis Result', result)

print("Нажмите любую клавишу для продолжения...")
cv2.waitKey(0)
cv2.destroyAllWindows()
