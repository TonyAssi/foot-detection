import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from huggingface_hub import hf_hub_download


class FootDetection:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_file = "fasterrcnn_foot.pth"
        self.model = self._load_model()
        self.last_detection = None

    def _load_model(self):
        local_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)

        # Download if not exists
        if not os.path.exists(local_path):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print("Downloading model from Hugging Face...")
            local_path = hf_hub_download(
                repo_id="tonyassi/foot-detection",
                filename=self.checkpoint_file,
                local_dir=self.checkpoint_dir
            )

        # Load model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model.load_state_dict(torch.load(local_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def detect(self, image, threshold=0.1):
        """Run foot detection on a PIL image"""
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        boxes = []
        scores = []
        for box, score in zip(outputs["boxes"], outputs["scores"]):
            if score >= threshold:
                boxes.append(box.tolist())
                scores.append(score.item())

        self.last_detection = {
            "boxes": boxes,
            "scores": scores
        }
        return self.last_detection

    def draw_boxes(self, image):
        """Draw the most recent detection boxes on a copy of the image"""
        if self.last_detection is None:
            raise ValueError("No detection results found. Run .detect(image) first.")
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        for box, score in zip(self.last_detection["boxes"], self.last_detection["scores"]):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0), f"{score:.2f}", fill="red")

        return image_copy
