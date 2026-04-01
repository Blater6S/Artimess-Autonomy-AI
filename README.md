# Artimess-Autonomy-AI
My First Mini Autonomous AI System


This project is a autonomous AI system that I’m building to work like a self-learning digital brain. The idea is to create something that can continuously observe, collect, process, and learn from data available on a computer, similar to how a brain learns from its environment.

The system scans different types of data such as text files, images, audio, and video, then converts that raw information into feature vectors (embeddings) and stores them in a local vector memory system for long-term learning.

At the center of the project is main.py, which acts as the controller. It runs two continuous loops: one for collecting and processing new data, and another for training the AI using stored memory.

The AI_data.py module handles file scanning, memory management, and system stability. It includes a homeostasis mechanism that adjusts scanning speed and workload based on CPU usage and system conditions, allowing the system to run continuously without overloading resources.

For multimodal understanding, the system uses separate modules:

AI_txt.py for text embeddings
AI_img.py for image analysis using CLIP, FaceNet, and MediaPipe
AI_voice.py for speech-to-text using Whisper
AI_net.py for collecting information from the internet and YouTube

The core brain is implemented in neuromodulated_plasticity.py, where multimodal data is combined using a fusion layer. It also includes a reinforcement learning-based system and a sleep-cycle-inspired mechanism that adjusts how the AI learns over time.

Overall, this project is an attempt to build a persistent, adaptive AI system that continuously learns, organizes knowledge, and improves itself over time.
