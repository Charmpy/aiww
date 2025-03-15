#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters
import torch
import numpy as np

class YOLOListener(Node):
    def __init__(self):
        super().__init__('yolo_listener')
        self.bridge = CvBridge()
        
        # Подписка на топики с использованием синхронизатора
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/depth_camera/image')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        
        # Загрузка модели YOLO11 (укажите корректный путь и настройки)
        self.model = torch.load('path_to_your_yolo11_model.pt', map_location=torch.device('cpu'))
        self.model.eval()
        
        # Пример значений параметров камеры – замените на реальные параметры вашей камеры
        self.fx = 525.0  # фокусное расстояние по x
        self.fy = 525.0  # фокусное расстояние по y
        self.cx = 319.5  # оптический центр по x
        self.cy = 239.5  # оптический центр по y

    def callback(self, rgb_msg, depth_msg):
        # Преобразование ROS-сообщений в изображения OpenCV
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Ошибка преобразования изображений: %s" % str(e))
            return
        
        # Преобразуем изображение для инференса (например, из BGR в RGB)
        input_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess_image(input_image)
        
        # Выполнение инференса модели (без градиентов)
        with torch.no_grad():
            predictions = self.model(input_tensor)[0]
        
        # Обработка результатов инференса – фильтрация боксов по порогу уверенности
        boxes = self.process_predictions(predictions)
        
        # Для каждого обнаруженного бокса вычисляем центр и преобразуем координаты в 3D
        for box in boxes:
            x1, y1, x2, y2, score, cls = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Получаем значение глубины в точке центра бокса
            depth = depth_image[center_y, center_x]
            if depth == 0:
                continue  # Пропускаем, если глубина не определена
            
            # Преобразование координат из пиксельных в декартовы
            X = (center_x - self.cx) * depth / self.fx
            Y = (center_y - self.cy) * depth / self.fy
            Z = depth
            
            # Отображение бокса и декартовых координат на изображении
            cv2.rectangle(rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(rgb_image, f'({X:.2f}, {Y:.2f}, {Z:.2f})',
                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Отображение изображения с боксов в реальном времени
        cv2.imshow("YOLO Detection", rgb_image)
        cv2.waitKey(1)

    def preprocess_image(self, image):
        """
        Преобразование изображения в формат, подходящий для модели:
          - нормализация
          - преобразование в тензор PyTorch
          - добавление batch размерности
        """
        image = image.astype(np.float32) / 255.0  # нормализация [0,1]
        image = np.transpose(image, (2, 0, 1))      # преобразование в формат C x H x W
        image = torch.from_numpy(image).unsqueeze(0)  # добавляем batch dimension
        return image

    def process_predictions(self, predictions):
        """
        Обработка вывода модели для получения списка боксов.
        Здесь предполагается, что predictions – это тензор, где каждая строка имеет формат:
        [x1, y1, x2, y2, score, class]
        Порог уверенности можно изменить по необходимости.
        """
        boxes = []
        conf_threshold = 0.5
        # Если predictions находится на GPU, переносим на CPU
        predictions = predictions.cpu() if predictions.is_cuda else predictions
        for pred in predictions:
            if pred[4] > conf_threshold:
                boxes.append(pred.numpy())
        return boxes

def main(args=None):
    rclpy.init(args=args)
    node = YOLOListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
