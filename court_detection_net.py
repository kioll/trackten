import cv2
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
from homography import get_trans_matrix, refer_kps

class CourtLineDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_keypoints(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0  # Scale x coordinates
        keypoints[1::2] *= original_h / 224.0  # Scale y coordinates
        return keypoints.reshape(-1, 2)  # Reshape for homography

    def infer_model(self, frames):
        kps_res = []
        matrixes_res = []
        for image in frames:
            keypoints = self.predict_keypoints(image)
            matrix_trans = get_trans_matrix(keypoints) if keypoints is not None else None
            kps_res.append(keypoints)
            matrixes_res.append(matrix_trans)
        return matrixes_res, kps_res

    def draw_keypoints(self, image, keypoints):
        for i in range(len(keypoints)):
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image
