# Author : P.P. Chanchal
import sys
import os
import asyncio
import logging
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Robust imports
try:
    import librosa
    import soundfile as sf
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    try:
        from pyannote.audio import Pipeline
    except Exception:
        Pipeline = None
except ImportError as e:
    logging.error(f"Critical dependency missing for AI_voice: {e}")
    # We continue, but functionality will be limited
    pass

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AI_VOICE] %(message)s'
)

class AuditoryCortex:
    """
    Advanced audio processor with continuous listening, 
    speech-to-text, and speaker diarization capabilities.
    """
    def __init__(self, device='cpu'):
        self.device = device
        logging.info(f"Initializing Auditory Cortex on {device}...")
        
        # 1. Whisper (Speech Recognition)
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
        except Exception as e:
            logging.error(f"Failed to load Whisper: {e}")
            self.model = None

        # 2. Pyannote (Speaker Diarization) - Optional
        self.diarization = None
        # Uncomment if you have a token
        # try:
        #     self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_TOKEN")
        # except Exception:
        #     pass

        self.sample_rate = 16000
        self.buffer = []
        self.is_listening = False

    async def process_audio_file(self, file_path: str) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Process an audio file to extract transcripts and speaker info.
        Returns: (transcripts_by_speaker, metadata)
        """
        if not self.model:
            return {"unknown": ["Model not loaded"]}, {}

        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Diarization (if available)
            speakers = self._diarize(file_path) if self.diarization else {}
            
            # Transcription
            # If no diarization, treat as single speaker "unknown"
            if not speakers:
                text = self._transcribe_segment(audio)
                return {"unknown": [text]}, {"duration": len(audio)/sr}
            
            # Segmented transcription
            results = {}
            for speaker, segments in speakers.items():
                results[speaker] = []
                for start, end in segments:
                    # Extract segment
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    if end_sample > len(audio): end_sample = len(audio)
                    
                    segment_audio = audio[start_sample:end_sample]
                    if len(segment_audio) > 0:
                        text = self._transcribe_segment(segment_audio)
                        if text:
                            results[speaker].append(text)
                            
            return results, {"duration": len(audio)/sr, "speaker_count": len(speakers)}

        except Exception as e:
            logging.error(f"Audio processing error for {file_path}: {e}")
            return {}, {}

    def _transcribe_segment(self, audio_segment: np.ndarray) -> str:
        try:
            input_features = self.processor(
                audio_segment, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""

    def _diarize(self, file_path: str) -> Dict[str, List[Tuple[float, float]]]:
        # Placeholder for real diarization logic
        # Returns { "SPEAKER_00": [(0.0, 5.0), (10.0, 15.0)] }
        return {}

    async def start_continuous_listening(self, duration=10, silence_threshold=0.01):
        """
        Listens to the microphone in chunks.
        """
        if not sd:
            logging.warning("sounddevice not available. Continuous listening disabled.")
            return

        logging.info("Started continuous listening...")
        self.is_listening = True
        
        while self.is_listening:
            try:
                # Record chunk
                recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
                sd.wait()
                
                audio = recording.flatten()
                
                # VAD (Voice Activity Detection) - Simple energy check
                if np.mean(np.abs(audio)) > silence_threshold:
                    logging.info("Sound detected, transcribing...")
                    text = self._transcribe_segment(audio)
                    if text:
                        logging.info(f"Heard: {text}")
                        # In a real system, we would push this to the memory manager
                        # For now, we just log it or yield it if this was a generator
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Listening error: {e}")
                await asyncio.sleep(1)

    def stop_listening(self):
        self.is_listening = False

# Wrapper for compatibility
class AutonomousAudioProcessor(AuditoryCortex):
    async def process_live_audio(self, duration=5):
        # One-shot listening for demo
        if not sd:
            print("Microphone not available.")
            return
            
        print(f"Listening for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        audio = recording.flatten()
        text = self._transcribe_segment(audio)
        print(f"Transcribed: {text}")
        return text

if __name__ == "__main__":
    # Test
    async def test():
        proc = AutonomousAudioProcessor()
        # Create dummy wav if needed, or just test init
        print("Audio processor initialized.")
        
    asyncio.run(test())
