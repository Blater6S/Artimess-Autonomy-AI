# Author : P.P. Chanchal
import os
import sys
import time
import json
import asyncio
import logging
import shutil
import psutil
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Union

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AI_DATA] %(message)s',
    handlers=[
        logging.FileHandler("ai_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class SystemHomeostasis:
    """
    Manages the 'health' and 'adaptation' of the AI system.
    Implements self-modification by adjusting configuration based on performance.
    """
    def __init__(self, config_path="system_state.json"):
        self.config_path = config_path
        self.state = self._load_state()
        self.history = []

    def _load_state(self) -> Dict:
        default_state = {
            "scan_interval": 10,  # Seconds
            "batch_size": 4,
            "plasticity_rate": 0.01,
            "energy_level": 100.0,
            "processed_count": 0,
            "error_count": 0,
            "last_active": time.time(),
            "adaptive_mode": True
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return {**default_state, **json.load(f)}
            except Exception as e:
                logging.error(f"Failed to load state: {e}")
                return default_state
        return default_state

    def save_state(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")

    def update_metrics(self, processed=0, errors=0):
        self.state["processed_count"] += processed
        self.state["error_count"] += errors
        self.state["last_active"] = time.time()
        
        # Self-Modification Logic: Adapt based on success/failure
        if self.state["adaptive_mode"]:
            if errors > 0:
                # Slow down if errors occur
                self.state["scan_interval"] = min(self.state["scan_interval"] * 1.5, 60)
                logging.warning(f"High errors detected. Increasing scan interval to {self.state['scan_interval']:.2f}s")
            elif processed > 0:
                # Speed up if processing is smooth
                self.state["scan_interval"] = max(self.state["scan_interval"] * 0.9, 1)
                
            # Energy management (simulated)
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > 80:
                self.state["energy_level"] -= 5
                logging.warning("High CPU usage. Reducing energy level.")
            else:
                self.state["energy_level"] = min(self.state["energy_level"] + 1, 100)

        self.save_state()

    def get_scan_interval(self):
        return self.state["scan_interval"]

class DriveScanner:
    """
    Advanced scanner that targets specific user folders on C: and full content on other drives.
    """
    def __init__(self):
        self.system = platform.system()
        self.user_home = Path.home()
        self.allowed_c_folders = ['Downloads', 'Documents', 'Pictures', 'Music', 'Videos']
        self.ignored_dirs = {'.git', '__pycache__', 'ai_env', 'venv', 'env', '.gemini', 'memory_vectors', '$RECYCLE.BIN', 'System Volume Information'}

    def get_search_paths(self) -> List[Path]:
        paths = []
        
        # 1. Add allowed C: folders
        for folder in self.allowed_c_folders:
            path = self.user_home / folder
            if path.exists():
                paths.append(path)
        
        # 2. Detect other drives (Windows specific)
        if self.system == "Windows":
            import string
            from ctypes import windll
            drives = []
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(letter)
                bitmask >>= 1
            
            for drive in drives:
                if drive != 'C':
                    drive_path = Path(f"{drive}:/")
                    if drive_path.exists():
                        paths.append(drive_path)
        
        return paths

    def scan_generator(self):
        """Yields file paths one by one from all valid sources"""
        search_paths = self.get_search_paths()
        logging.info(f"Scanning the following roots: {[str(p) for p in search_paths]}")
        
        for root_path in search_paths:
            if not root_path.exists():
                continue
                
            try:
                for root, dirs, files in os.walk(root_path):
                    # In-place modification to prune ignored directories
                    dirs[:] = [d for d in dirs if d not in self.ignored_dirs and not d.startswith('.')]
                    
                    for file in files:
                        if file.startswith('.'): continue
                        yield Path(root) / file
            except PermissionError:
                logging.warning(f"Permission denied accessing {root_path}")
            except Exception as e:
                logging.error(f"Error scanning {root_path}: {e}")

class VectorMemoryManager:
    """
    Manages the storage and retrieval of vector embeddings.
    """
    def __init__(self, base_dir="memory_vectors"):
        self.base_dir = Path(base_dir)
        self.metadata_file = self.base_dir / "metadata.json"
        self.setup_storage()
        self.metadata = self.load_metadata()

    def setup_storage(self):
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "text").mkdir(exist_ok=True)
        (self.base_dir / "image").mkdir(exist_ok=True)
        (self.base_dir / "audio").mkdir(exist_ok=True)

    def load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    # Check if it's the old format (keys are modalities)
                    if "text" in data and isinstance(data["text"], list):
                        logging.warning("Old metadata format detected. Resetting metadata.")
                        return {}
                    return data
            except Exception:
                return {}
        return {}

    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_vector(self, vector, file_path: str, modality: str, extra_info: Dict = None):
        import numpy as np
        import torch
        
        file_hash = str(hash(file_path)) # Simple hash for demo
        timestamp = datetime.now().isoformat()
        
        # Convert to numpy for storage
        if isinstance(vector, torch.Tensor):
            vec_np = vector.detach().cpu().numpy()
        elif isinstance(vector, list):
            vec_np = np.array(vector)
        else:
            vec_np = vector
            
        save_path = self.base_dir / modality / f"{file_hash}.npy"
        np.save(save_path, vec_np)
        
        self.metadata[file_hash] = {
            "original_path": str(file_path),
            "modality": modality,
            "timestamp": timestamp,
            "vector_path": str(save_path),
            "shape": vec_np.shape,
            "info": extra_info or {}
        }
        self.save_metadata()
        return file_hash

    def is_processed(self, file_path: str) -> bool:
        # Check if file is already in metadata (naive check by path)
        # In a real system, use content hash
        for entry in self.metadata.values():
            if entry["original_path"] == str(file_path):
                return True
        return False

class AutonomousDataManager:
    """
    The central coordinator that runs continuously.
    """
    def __init__(self):
        self.homeostasis = SystemHomeostasis()
        self.scanner = DriveScanner()
        self.memory = VectorMemoryManager()
        self.running = True
        
        # Queues for processing (could be used for async workers)
        self.processing_queue = asyncio.Queue()

    async def ingest_file(self, file_path: Path, processors: Dict):
        """Route file to appropriate processor"""
        if self.memory.is_processed(str(file_path)):
            return False

        ext = file_path.suffix.lower()
        processed = False
        
        try:
            if ext in ['.txt', '.csv', '.xlsx', '.pdf', '.docx', '.pptx']:
                if 'text' in processors:
                    vec = await processors['text'].extract_vector(file_path)
                    self.memory.save_vector(vec, str(file_path), 'text')
                    processed = True
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                if 'image' in processors:
                    vec, info = await processors['image'].extract_features(str(file_path))
                    self.memory.save_vector(vec, str(file_path), 'image', {'expressions': info})
                    processed = True
                    
            elif ext in ['.wav', '.mp3', '.flac']:
                if 'audio' in processors:
                    transcripts, _ = await processors['audio'].process_audio_file(str(file_path))
                    # Flatten transcript
                    full_text = " ".join([t for sublist in transcripts.values() for t in sublist])
                    if full_text and 'text' in processors:
                        vec = await processors['text']._process_text(full_text)
                        self.memory.save_vector(vec, str(file_path), 'audio', {'transcript': full_text})
                        processed = True
                        
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            self.homeostasis.update_metrics(errors=1)
            return False

        if processed:
            logging.info(f"Successfully ingested: {file_path}")
            self.homeostasis.update_metrics(processed=1)
            
        return processed

    async def start_autonomous_loop(self, processors: Dict):
        logging.info("Starting Autonomous Data Loop...")
        
        while self.running:
            try:
                interval = self.homeostasis.get_scan_interval()
                logging.info(f"Scanning cycle started. Next scan in {interval}s")
                
                count = 0
                # Use the generator to scan one by one
                for file_path in self.scanner.scan_generator():
                    if not self.running: break
                    
                    # Process file
                    await self.ingest_file(file_path, processors)
                    
                    # Yield to event loop occasionally to prevent freezing
                    count += 1
                    if count % 10 == 0:
                        await asyncio.sleep(0.01)
                
                logging.info("Scan cycle complete.")
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logging.info("Loop cancelled.")
                break
            except Exception as e:
                logging.error(f"Critical loop error: {e}")
                await asyncio.sleep(5) # Safety backoff

    def stop(self):
        self.running = False

if __name__ == "__main__":
    # Test stub
    mgr = AutonomousDataManager()
    paths = mgr.scanner.get_search_paths()
    print(f"Found search paths: {paths}")
    print("Homeostasis state:", mgr.homeostasis.state)
