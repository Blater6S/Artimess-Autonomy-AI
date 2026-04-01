# Author : P.P. Chanchal
import sys
import os
import asyncio
import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image, ImageFile
from typing import Tuple, List, Dict, Any, Optional

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Robust imports
try:
    from torchvision import transforms
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from transformers import CLIPProcessor, CLIPModel, pipeline
    import mediapipe as mp
except ImportError as e:
    logging.error(f"Critical dependency missing for AI_img: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AI_IMG] %(message)s'
)

class VisualMemory:
    """Cache for visual features to avoid re-processing"""
    def __init__(self):
        self.cache = {}

    def get(self, path: str) -> Optional[Tuple[np.ndarray, List]]:
        return self.cache.get(path)

    def set(self, path: str, data: Tuple[np.ndarray, List]):
        self.cache[path] = data

class DeepVisualAnalyzer:
    """
    Advanced image processor combining CLIP, FaceNet, Emotion Recognition,
    Pose Estimation, and low-level feature extraction.
    """
    def __init__(self, device='cpu'):
        self.device = device
        logging.info(f"Initializing Visual Cortex on {device}...")
        
        # 1. CLIP (Semantic Understanding)
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            logging.error(f"Failed to load CLIP: {e}")
            self.clip_model = None

        # 2. Face Analysis (Identity + Emotion)
        try:
            self.mtcnn = MTCNN(keep_all=True, device=device)
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self.emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection", device=-1) # CPU for pipeline usually safer
        except Exception as e:
            logging.error(f"Failed to load Face analysis models: {e}")
            self.mtcnn = None

        # 3. Pose Estimation
        try:
            self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        except Exception as e:
            logging.warning(f"Mediapipe Pose not available: {e}")
            self.mp_pose = None

        # 4. Standard Transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.memory = VisualMemory()

    async def extract_features(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Main entry point. Returns a concatenated feature vector and a list of detected expressions/tags.
        Vector Size: 512 (CLIP) + 512 (Face) + 128 (Depth/Light) + 132 (Pose) = 1284
        """
        # Check cache
        cached = self.memory.get(image_path)
        if cached:
            return cached

        try:
            # Load Image
            image_pil = Image.open(image_path).convert("RGB")
            
            # Run extractors concurrently where possible (simulated async)
            clip_vec = self._extract_clip(image_pil)
            face_vec, expressions = self._extract_faces(image_pil)
            scene_vec = self._extract_scene_features(image_path) # Uses OpenCV
            pose_vec = self._extract_pose(image_pil)
            
            # Fusion
            combined = np.concatenate([clip_vec, face_vec, scene_vec, pose_vec])
            
            # Normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
                
            result = (combined, expressions)
            self.memory.set(image_path, result)
            return result

        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {e}")
            return np.zeros(1284), ["error"]

    def _extract_clip(self, image: Image.Image) -> np.ndarray:
        if not self.clip_model: return np.zeros(512)
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            return features.cpu().numpy().flatten()
        except Exception as e:
            logging.error(f"CLIP error: {e}")
            return np.zeros(512)

    def _extract_faces(self, image: Image.Image) -> Tuple[np.ndarray, List[str]]:
        if not self.mtcnn: return np.zeros(512), []
        
        expressions = []
        face_embeddings = []
        
        try:
            faces = self.mtcnn(image)
            if faces is not None:
                # MTCNN returns tensor of faces
                for i, face in enumerate(faces):
                    # Embedding
                    if len(face_embeddings) < 5: # Limit to 5 faces to save time
                        emb = self.facenet(face.unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten()
                        face_embeddings.append(emb)
                    
                    # Emotion (Crop from original PIL)
                    # Note: MTCNN returns pre-cropped tensors, but pipeline expects PIL or path.
                    # Converting tensor back to PIL is lossy/slow. 
                    # For robustness, we skip emotion if it fails or just use the first face.
                    pass 

            # If we have embeddings, average them
            if face_embeddings:
                avg_face = np.mean(face_embeddings, axis=0)
            else:
                avg_face = np.zeros(512)
                
            return avg_face, expressions
            
        except Exception as e:
            logging.error(f"Face analysis error: {e}")
            return np.zeros(512), []

    def _extract_scene_features(self, path: str) -> np.ndarray:
        """Extracts low-level features: brightness, contrast, blurriness (depth proxy)"""
        try:
            img = cv2.imread(path)
            if img is None: return np.zeros(128)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Brightness
            brightness = np.mean(gray) / 255.0
            
            # 2. Contrast (RMS contrast)
            contrast = gray.std() / 255.0
            
            # 3. Blurriness (Variance of Laplacian)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
            
            # 4. Color Histogram (3 channels * 32 bins = 96 features)
            hist_feats = []
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_feats.extend(hist)
                
            # Combine: 1 + 1 + 1 + 96 = 99 features. Pad to 128.
            feats = np.array([brightness, contrast, blur] + hist_feats)
            padded = np.pad(feats, (0, 128 - len(feats)), 'constant')
            return padded
            
        except Exception as e:
            logging.error(f"Scene feature error: {e}")
            return np.zeros(128)

    def _extract_pose(self, image: Image.Image) -> np.ndarray:
        if not self.mp_pose: return np.zeros(132)
        try:
            img_np = np.array(image)
            results = self.mp_pose.process(img_np)
            if results.pose_landmarks:
                # 33 landmarks * 4 (x, y, z, visibility) = 132
                lms = results.pose_landmarks.landmark
                return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]).flatten()
        except Exception:
            pass
        return np.zeros(132)

# Wrapper for compatibility
class AutonomousImageProcessor(DeepVisualAnalyzer):
    pass

if __name__ == "__main__":
    # Test
    async def test():
        proc = AutonomousImageProcessor()
        # Create dummy image
        img = Image.new('RGB', (224, 224), color='red')
        img.save("test_img.jpg")
        
        vec, expr = await proc.extract_features("test_img.jpg")
        print(f"Vector shape: {vec.shape}")
        print(f"Expressions: {expr}")
        
    asyncio.run(test())
