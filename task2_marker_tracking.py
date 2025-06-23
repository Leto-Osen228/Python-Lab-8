import cv2
import numpy as np

# Загружаем изображение метки (шаблона)
template = cv2.imread('Python-Lab-8/ref-point.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError('ref-point.jpg не найден!')

cv2.imshow('Template (ref-point)', template)

# Загружаем изображение мухи
fly_img = cv2.imread('Python-Lab-8/fly64.png', cv2.IMREAD_UNCHANGED)
if fly_img is None:
    raise FileNotFoundError('fly64.png не найден!')

# Получаем размеры шаблона
template_h, template_w = template.shape

def preprocess_image(img):
    """Предобработка изображения для улучшения сопоставления"""
    # Применяем гауссово размытие для уменьшения шума
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # Нормализация гистограммы для улучшения контраста
    normalized = cv2.equalizeHist(blurred)
    return normalized

def overlay_fly_on_frame(frame, center, scale):
    """
    Накладывает изображение мухи на кадр с центром в указанной точке
    """
    if fly_img is None:
        return frame
    
    # Масштабируем муху в зависимости от масштаба метки
    fly_scale = max(0.3, min(2.0, scale))  # Ограничиваем масштаб от 0.3 до 2.0
    new_fly_width = int(fly_img.shape[1] * fly_scale)
    new_fly_height = int(fly_img.shape[0] * fly_scale)
    
    if new_fly_width <= 0 or new_fly_height <= 0:
        return frame
    
    # Изменяем размер мухи
    resized_fly = cv2.resize(fly_img, (new_fly_width, new_fly_height))
    
    # Вычисляем позицию для наложения (центр мухи = центр метки)
    x_offset = center[0] - new_fly_width // 2
    y_offset = center[1] - new_fly_height // 2
    
    # Проверяем границы кадра
    frame_h, frame_w = frame.shape[:2]
    
    # Обрезаем муху, если она выходит за границы кадра
    fly_x_start = max(0, -x_offset)
    fly_y_start = max(0, -y_offset)
    fly_x_end = min(new_fly_width, frame_w - x_offset)
    fly_y_end = min(new_fly_height, frame_h - y_offset)
    
    frame_x_start = max(0, x_offset)
    frame_y_start = max(0, y_offset)
    frame_x_end = min(frame_w, x_offset + new_fly_width)
    frame_y_end = min(frame_h, y_offset + new_fly_height)
    
    if fly_x_end <= fly_x_start or fly_y_end <= fly_y_start:
        return frame
    
    # Извлекаем область мухи для наложения
    fly_region = resized_fly[fly_y_start:fly_y_end, fly_x_start:fly_x_end]
    
    # Если муха имеет альфа-канал (прозрачность)
    if resized_fly.shape[2] == 4:
        # Разделяем на BGR и альфа-канал
        fly_bgr = fly_region[:, :, :3]
        fly_alpha = fly_region[:, :, 3] / 255.0
        
        # Получаем область кадра для наложения
        frame_region = frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
        
        # Применяем альфа-смешивание
        for c in range(3):
            frame_region[:, :, c] = (fly_alpha * fly_bgr[:, :, c] + 
                                   (1 - fly_alpha) * frame_region[:, :, c])
        
        frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end] = frame_region
    else:
        # Простое наложение без прозрачности
        frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end] = fly_region
    
    return frame

def find_marker_template_matching(frame):
    """
    Поиск маркера с помощью template matching с разными масштабами и поворотами
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_processed = preprocess_image(gray)
    template_processed = preprocess_image(template)
    
    best_match = None
    best_confidence = 0.5  # Понижен порог для лучшего обнаружения
    
    # Больше масштабов для работы на разных расстояниях
    scales = np.linspace(0.3, 3.0, 20)
    # Углы поворота (каждые 90 градусов)
    angles = range(0, 180, 45)
    
    for scale in scales:
        # Изменяем размер шаблона
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)
        
        if new_w < 15 or new_h < 15 or new_w > gray.shape[1] or new_h > gray.shape[0]:
            continue
        
        for angle in angles:
            # Поворачиваем шаблон
            center_template = (template_w // 2, template_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center_template, angle, scale)
            
            # Вычисляем новые размеры после поворота
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            
            new_w_rot = int((template_h * sin_angle) + (template_w * cos_angle))
            new_h_rot = int((template_h * cos_angle) + (template_w * sin_angle))
            
            if new_w_rot > gray.shape[1] or new_h_rot > gray.shape[0] or new_w_rot < 15 or new_h_rot < 15:
                continue
            
            # Корректируем центр поворота
            rotation_matrix[0, 2] += (new_w_rot / 2) - center_template[0]
            rotation_matrix[1, 2] += (new_h_rot / 2) - center_template[1]
            
            # Применяем поворот и масштабирование
            rotated_template = cv2.warpAffine(template_processed, rotation_matrix, (new_w_rot, new_h_rot))
            
            # Выполняем template matching
            if rotated_template.shape[0] > 0 and rotated_template.shape[1] > 0:
                result = cv2.matchTemplate(gray_processed, rotated_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    top_left = max_loc
                    bottom_right = (top_left[0] + new_w_rot, top_left[1] + new_h_rot)
                    center = (top_left[0] + new_w_rot // 2, top_left[1] + new_h_rot // 2)
                    best_match = {
                        'center': center,
                        'top_left': top_left,
                        'bottom_right': bottom_right,
                        'confidence': max_val,
                        'size': (new_w_rot, new_h_rot),
                        'angle': angle,
                        'scale': scale
                    }
    
    return best_match

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break    # Поиск маркера
    match = find_marker_template_matching(frame)
    
    if match is not None:
        center = match['center']
        top_left = match['top_left']
        bottom_right = match['bottom_right']
        confidence = match['confidence']
        angle = match['angle']
        scale = match['scale']
        
        # Определяем цвет обводки в зависимости от положения маркера
        frame_height, frame_width = frame.shape[:2]
        
        # Проверяем положение центра метки
        if center[0] <= 150 and center[1] <= 150:
            # Левый верхний угол - синий цвет
            color = (255, 0, 0)  # BGR: синий
            position_text = "Top-Left (Blue)"
        elif center[0] >= frame_width - 150 and center[1] >= frame_height - 150:
            # Правый нижний угол - красный цвет
            color = (0, 0, 255)  # BGR: красный
            position_text = "Bottom-Right (Red)"
        else:
            # Остальные области - зелёный цвет
            color = (0, 255, 0)  # BGR: зелёный
            position_text = "Normal (Green)"
          # Рисуем найденный маркер с соответствующим цветом
        cv2.rectangle(frame, top_left, bottom_right, color, 3)
        cv2.circle(frame, center, 5, color, -1)
        
        # Накладываем изображение мухи на центр метки
        frame = overlay_fly_on_frame(frame, center, scale*4)
        
        # Рисуем области для визуализации зон
        cv2.rectangle(frame, (0, 0), (150, 150), (255, 0, 0), 2)  # Синяя зона
        cv2.rectangle(frame, (frame_width - 150, frame_height - 150), (frame_width, frame_height), (0, 0, 255), 2)  # Красная зона

        cv2.putText(frame, f'Marker: {center}', (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f'Position: {position_text}', (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f'Angle: {angle}°, Scale: {scale:.2f}', (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f'Confidence: {confidence:.3f}', (10, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Marker Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc для выхода
        break

cap.release()
cv2.destroyAllWindows()
