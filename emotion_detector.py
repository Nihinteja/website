"""
=============================================================================
AI-Based Emotion Detection System
Module: emotion_detector.py
Description: Core Facial Detection & Deep Learning Emotion Recognition Engine
=============================================================================
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONSTANTS & METADATA
# =============================================================================

EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

EMOTION_DISPLAY_NAMES: Dict[str, str] = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}

EMOTION_ICONS: Dict[str, str] = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# Color palette for bounding boxes and UI badges (BGR for OpenCV, Hex for CSS)
EMOTION_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "angry": (50, 50, 235),       # Crimson Red
    "disgust": (34, 139, 34),     # Forest Green
    "fear": (180, 50, 160),      # Deep Purple / Magenta
    "happy": (40, 200, 80),      # Bright Emerald
    "sad": (220, 120, 40),       # Cobalt Blue
    "surprise": (30, 200, 240),   # Vibrant Amber / Gold
    "neutral": (180, 180, 180),   # Slate Grey
}

EMOTION_COLORS_HEX: Dict[str, str] = {
    "angry": "#ef4444",
    "disgust": "#22c55e",
    "fear": "#a855f7",
    "happy": "#10b981",
    "sad": "#3b82f6",
    "surprise": "#f59e0b",
    "neutral": "#94a3b8",
}


# =============================================================================
# DEEP LEARNING ARCHITECTURE (Mini-Xception for FER-2013)
# =============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution for lightweight, fast inference."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MiniXception(nn.Module):
    """
    Mini-Xception Deep Residual CNN trained on FER-2013.
    Recognizes 7 facial expressions from 64x64 grayscale face crops.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        # Initial Convolutional Layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)

        # Residual Block 1 (8 -> 16 channels)
        self.res1 = nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn_res1 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.sep1 = SeparableConv2d(8, 16, 3, padding=1)
        self.bn_sep1 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.sep2 = SeparableConv2d(16, 16, 3, padding=1)
        self.bn_sep2 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Block 2 (16 -> 32 channels)
        self.res2 = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn_res2 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.sep3 = SeparableConv2d(16, 32, 3, padding=1)
        self.bn_sep3 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.sep4 = SeparableConv2d(32, 32, 3, padding=1)
        self.bn_sep4 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Block 3 (32 -> 64 channels)
        self.res3 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn_res3 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.sep5 = SeparableConv2d(32, 64, 3, padding=1)
        self.bn_sep5 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.sep6 = SeparableConv2d(64, 64, 3, padding=1)
        self.bn_sep6 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Block 4 (64 -> 128 channels)
        self.res4 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn_res4 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.sep7 = SeparableConv2d(64, 128, 3, padding=1)
        self.bn_sep7 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.sep8 = SeparableConv2d(128, 128, 3, padding=1)
        self.bn_sep8 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Classification Head
        self.conv_head = nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Block 1
        res = self.bn_res1(self.res1(x))
        x = F.relu(self.bn_sep1(self.sep1(x)))
        x = self.pool1(self.bn_sep2(self.sep2(x)))
        x = x + res

        # Block 2
        res = self.bn_res2(self.res2(x))
        x = F.relu(self.bn_sep3(self.sep3(x)))
        x = self.pool2(self.bn_sep4(self.sep4(x)))
        x = x + res

        # Block 3
        res = self.bn_res3(self.res3(x))
        x = F.relu(self.bn_sep5(self.sep5(x)))
        x = self.pool3(self.bn_sep6(self.sep6(x)))
        x = x + res

        # Block 4
        res = self.bn_res4(self.res4(x))
        x = F.relu(self.bn_sep7(self.sep7(x)))
        x = self.pool4(self.bn_sep8(self.sep8(x)))
        x = x + res

        # Global average pooling & Softmax
        x = self.conv_head(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.softmax(x, dim=-1)
        return x


# =============================================================================
# FACIAL EMOTION DETECTOR ENGINE
# =============================================================================

class EmotionDetector:
    """
    Production-grade Facial Emotion Detection Engine.
    Detects faces in images/video and predicts emotion distributions across 7 classes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        cascade_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Resolve base directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Set up face cascade
        if cascade_path is None:
            local_cascade = os.path.join(self.base_dir, "models", "haarcascade_frontalface_default.xml")
            if os.path.exists(local_cascade):
                cascade_path = local_cascade
            else:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from: {cascade_path}")

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set up deep learning model
        if model_path is None:
            model_path = os.path.join(self.base_dir, "models", "emotion_model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained emotion recognition model weights not found at: {model_path}")

        self.model = MiniXception(num_classes=7)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.input_size = (64, 64)

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
    ) -> List[Tuple[int, int, int, int]]:
        """Detect face bounding boxes (x, y, w, h) in a BGR or RGB image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Enhance contrast for better detection under varied lighting
        equalized = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size,
        )

        return [tuple(map(int, face)) for face in faces]

    def preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a face crop to match the FER-2013 training pipeline:
        Grayscale -> 64x64 -> Normalized to [-1.0, 1.0] -> Tensor (1, 1, 64, 64).
        """
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) * 2.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def predict_emotion(self, face_crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion probabilities for a single face crop.
        Returns:
            top_emotion (str): Highest scoring emotion key
            confidence (float): Confidence percentage (0.0 to 100.0)
            scores (dict): Dictionary mapping emotion names to percentage scores
        """
        tensor = self.preprocess_face(face_crop)
        with torch.no_grad():
            preds = self.model(tensor).cpu().numpy()[0]

        scores = {EMOTIONS[i]: float(preds[i] * 100.0) for i in range(len(EMOTIONS))}
        top_idx = int(np.argmax(preds))
        top_emotion = EMOTIONS[top_idx]
        top_confidence = float(preds[top_idx] * 100.0)

        return top_emotion, top_confidence, scores

    def process_image(
        self,
        image: np.ndarray,
        draw_annotations: bool = True,
        box_thickness: int = 2,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes an input image: detects faces, classifies emotions, and draws modern overlays.

        Args:
            image (np.ndarray): Input BGR image
            draw_annotations (bool): Whether to draw bounding boxes and labels
            box_thickness (int): Line thickness for annotations

        Returns:
            annotated_image (np.ndarray): Image with styled overlays
            detections (list): List of detected face metadata dicts
        """
        output_image = image.copy()
        h_img, w_img = image.shape[:2]
        faces = self.detect_faces(image)
        detections: List[Dict[str, Any]] = []

        for (x, y, w, h) in faces:
            # Add padding around face crop for better context
            pad_x = int(0.1 * w)
            pad_y = int(0.1 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            top_emotion, confidence, scores = self.predict_emotion(face_crop)
            display_name = EMOTION_DISPLAY_NAMES[top_emotion]
            color = EMOTION_COLORS_BGR.get(top_emotion, (0, 255, 0))

            face_record = {
                "box": (x, y, w, h),
                "emotion": top_emotion,
                "display_name": display_name,
                "confidence": confidence,
                "scores": scores,
                "face_crop": face_crop,
                "icon": EMOTION_ICONS[top_emotion],
                "color_hex": EMOTION_COLORS_HEX[top_emotion],
            }
            detections.append(face_record)

            if draw_annotations:
                # 1. Sleek Bounding Box with Accented Corner Brackets
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, box_thickness)
                corner_len = max(10, int(min(w, h) * 0.15))
                c_thick = box_thickness + 2

                # Top-Left
                cv2.line(output_image, (x, y), (x + corner_len, y), color, c_thick)
                cv2.line(output_image, (x, y), (x, y + corner_len), color, c_thick)
                # Top-Right
                cv2.line(output_image, (x + w, y), (x + w - corner_len, y), color, c_thick)
                cv2.line(output_image, (x + w, y), (x + w, y + corner_len), color, c_thick)
                # Bottom-Left
                cv2.line(output_image, (x, y + h), (x + corner_len, y + h), color, c_thick)
                cv2.line(output_image, (x, y + h), (x, y + h - corner_len), color, c_thick)
                # Bottom-Right
                cv2.line(output_image, (x + w, y + h), (x + w - corner_len, y + h), color, c_thick)
                cv2.line(output_image, (x + w, y + h), (x + w, y + h - corner_len), color, c_thick)

                # 2. Modern Tag Pill Above Bounding Box
                label_text = f"{display_name} {confidence:.1f}%"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = max(0.45, min(0.75, w / 220.0))
                font_thick = 1

                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
                pill_h = text_h + 12
                pill_w = text_w + 14

                pill_y1 = max(0, y - pill_h - 4)
                pill_y2 = pill_y1 + pill_h
                pill_x1 = x
                pill_x2 = min(w_img, x + pill_w)

                # Background rounded badge
                cv2.rectangle(output_image, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
                # Text inside badge
                text_x = pill_x1 + 7
                text_y = pill_y1 + text_h + 4
                cv2.putText(
                    output_image,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thick,
                    cv2.LINE_AA,
                )

        return output_image, detections
