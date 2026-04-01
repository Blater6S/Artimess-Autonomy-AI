# Author : P.P. Chanchal
import asyncio
import os
import sys
import logging
import signal
import importlib.util
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [MAIN] %(message)s',
    handlers=[
        logging.FileHandler("system_main.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Dynamic Import Helper
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import Modules
try:
    from AI_data import AutonomousDataManager
    from AI_txt import UniversalFileProcessor
    from AI_img import AutonomousImageProcessor
    from AI_voice import AutonomousAudioProcessor
    
    # Import Brain
    neuro_plasticity = import_module_from_path("neuromodulated_plasticity", "neuromodulated plasticity.py")
    AutonomousNeuromodulatedAI = neuro_plasticity.AutonomousNeuromodulatedAI
    
except ImportError as e:
    logging.critical(f"Failed to import core modules: {e}")
    sys.exit(1)

class AutonomousSystem:
    """
    The Main Orchestrator.
    Initializes all subsystems and manages the infinite life cycle.
    """
    def __init__(self):
        self.running = True
        self.data_manager = AutonomousDataManager()
        self.brain = AutonomousNeuromodulatedAI()
        
        # Processors
        self.processors = {
            'text': UniversalFileProcessor(),
            'image': AutonomousImageProcessor(),
            'audio': AutonomousAudioProcessor()
        }
        
        # Handle Signals
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logging.info("Shutdown signal received. Stopping system...")
        self.running = False
        self.data_manager.stop()

    async def run(self):
        logging.info("=== Starting Autonomous AI System ===")
        logging.info("Mode: Fully Autonomous | Self-Modifying | Continuous")
        
        # 1. Start Data Ingestion Loop (Background)
        ingestion_task = asyncio.create_task(
            self.data_manager.start_autonomous_loop(self.processors)
        )
        
        # 2. Start Brain/Training Loop (Foreground)
        logging.info("Starting Cognitive Loop...")
        
        step = 0
        while self.running:
            try:
                # Fetch data from memory (Simulated retrieval for training)
                # In a real system, we'd query the VectorMemoryManager
                # Here we just iterate through what we have
                
                # Check if we have any data
                if not self.data_manager.memory.metadata:
                    logging.info("Waiting for data ingestion...")
                    await asyncio.sleep(5)
                    continue
                
                # Pick a random memory to train on (Replay)
                import random
                keys = list(self.data_manager.memory.metadata.keys())
                key = random.choice(keys)
                entry = self.data_manager.memory.metadata[key]
                
                # Load vector
                import numpy as np
                import torch
                try:
                    vec_np = np.load(entry['vector_path'])
                    vector = torch.from_numpy(vec_np).float().unsqueeze(0)
                    
                    # Train Step
                    status = self.brain.train_step(vector, vector)
                    
                    if step % 10 == 0:
                        logging.info(f"Step {step} | Action: {status['action']} | Loss: {status['avg_loss']:.4f}")
                        
                    step += 1
                    
                except Exception as e:
                    logging.error(f"Training error on {key}: {e}")
                
                # Yield
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Cognitive loop error: {e}")
                await asyncio.sleep(1)

        # Cleanup
        logging.info("Waiting for ingestion to stop...")
        await ingestion_task
        logging.info("System Shutdown Complete.")

if __name__ == "__main__":
    system = AutonomousSystem()
    asyncio.run(system.run())
