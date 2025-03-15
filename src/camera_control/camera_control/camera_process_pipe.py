import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters
import torch
import numpy as np
import os
import torch.serialization
from ultralytics import YOLO
class YOLOListener(Node):
    def __init__(self):
        super().__init__('yolo_listener')
        self.bridge = CvBridge()
        self.counter = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/depth_camera/image')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        # Number of classes (e.g., 80 for COCO)
        num_classes = 6
        self.color_palette = {}
        for i in range(num_classes):
            # Generate distinct colors using simple arithmetic.
            # These factors (37, 17, 29) are arbitrary; you can experiment with different values.
            r = (i * 50) % 255
            g = (i * 2) % 255
            b = (i * 13) % 255
            self.color_palette[i] = (b, g, r)  # OpenCV uses BGR format

        print(f"DEBUG: cwd: {os.getcwd()}")
        # with torch.serialization.safe_globals([DetectionModel]):
        #     checkpoint = torch.load('trainedNN/yolo_detect.pt', map_location=self.device)

        # # Check if the checkpoint is a dictionary with a 'model' key.
        # if isinstance(checkpoint, dict) and 'model' in checkpoint:
        #     self.model = checkpoint['model']
        # else:
        #     self.model = checkpoint

        # self.model.to(self.device)
        # self.model.eval()
        
        self.model = YOLO("trainedNN/yolo_detect.pt")
        self.fx = 525.0
        self.fy = 525.0  
        self.cx = 319.5  
        self.cy = 239.5  

    def callback(self, rgb_msg, depth_msg):
        
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Ошибка преобразования изображений: %s" % str(e))
            return
        
        
        # Process the model's output and draw bounding boxes with class-specific colors and larger text
        results = self.model(rgb_image)
        for r in results:
            # Convert bounding boxes and class indices to numpy arrays
            boxes = r.boxes.xyxy.cpu().numpy()   # shape: (N, 4)
            cls_ids = r.boxes.cls.cpu().numpy()    # shape: (N,)
            
            # Iterate over each detection using zip
            for box, cls_id in zip(boxes, cls_ids):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Get minimum depth in the bounding box area using your custom function
                min_d = self.traversebb_depth(int(x1), int(y1), int(x2), int(y2), depth_image)
                
                # Retrieve the class-specific color from the palette
                color = self.color_palette.get(int(cls_id), (0, 255, 0))
                
                # Draw bounding box with a thicker line
                cv2.rectangle(rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                
                # Overlay depth information at the center of the box with larger text
                cv2.putText(
                    rgb_image, f'{min_d:.2f}',
                    (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
                )
                
                # Retrieve the class name using the class id from r.names
                class_name = r.names[int(cls_id)]
                # Overlay the class name near the top-left corner of the bounding box with larger text
                cv2.putText(
                    rgb_image, f'{class_name}',
                    (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
                )

        # input_tensor = self.preprocess_image(input_image)
        
        # input_tensor = input_tensor.to(self.device)
        
        # if self.device.type == 'cuda':
        #     input_tensor = input_tensor.half()

        # with torch.no_grad():
        #     predictions = self.model(input_tensor)[0]
        
        
        # boxes = self.process_predictions(predictions)
        
        # for box in boxes:
        #     x1, y1, x2, y2, score, cls = box
        #     print(f"Found smth\n")
        #     center_x = int((x1 + x2) / 2)
        #     center_y = int((y1 + y2) / 2)
            
            
        #     depth = depth_image[center_y, center_x]
        #     if depth == 0:
        #         continue  
            
            
        #     X = (center_x - self.cx) * depth / self.fx
        #     Y = (center_y - self.cy) * depth / self.fy
        #     Z = depth
            
            
        #     cv2.rectangle(rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #     cv2.putText(
        #         rgb_image, f'({X:.2f}, {Y:.2f}, {Z:.2f})',
        #         (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        #     )
        
        self.counter+=1
        cv2.imshow("YOLO Detection", rgb_image)
        cv2.imwrite(f'artifacts/img{self.counter}.png', rgb_image)
        cv2.waitKey(1)


    def traversebb_depth(self, x1, y1, x2, y2, depth_image):
        min_depth = 1e9
        for y in range(y1,y2):
            for x in range(x1,x2):
                if depth_image[y,x] <= 0:
                    continue
                min_depth = min(min_depth, depth_image[y,x])
        return min_depth

    def preprocess_image(self, image):
        """
        Преобразование изображения в формат, подходящий для модели:
          - нормализация
          - преобразование в тензор PyTorch
          - добавление batch размерности
        """
        image = image.astype(np.float32) / 255.0  
        image = np.transpose(image, (2, 0, 1))      
        image = torch.from_numpy(image).unsqueeze(0)  
        return image

    def process_predictions(self, predictions):
        boxes = []
        conf_threshold = 0.49
        # predictions shape is (1, 84, 8400)
        # Remove batch dimension: now shape (84, 8400)
        predictions = predictions.squeeze(0)
        # Transpose so each row is one prediction: shape (8400, 84)
        predictions = predictions.t()
        
        # Iterate over each grid prediction
        for pred in predictions:
            # First 4 elements: bbox (center_x, center_y, width, height)
            bbox = pred[0:4]
            # 5th element: object confidence score
            obj_conf = pred[4].item()
            # Remaining 80 elements: class probabilities
            class_probs = pred[5:]
            # Identify best class probability and index
            best_class_prob, best_class_idx = torch.max(class_probs, 0)
            best_class_prob = best_class_prob.item()
            best_class_idx = best_class_idx.item()
            # Calculate the final detection score
            score = obj_conf * best_class_prob

            if score > conf_threshold:
                print('found smth')
                x1 = bbox[0].item()
                y1 = bbox[1].item()
                w  = bbox[2].item()
                h  = bbox[3].item()
                # Convert from center format (cx, cy, w, h) to (x1, y1, x2, y2)
                x2 = x1 + w
                y2 = y1 + h
                boxes.append([x1, y1, x2, y2, score, best_class_idx])
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
