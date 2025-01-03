from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np
import logging
import threading
import queue
import time
import torch
import scipy.signal
import os
from pathlib import Path
from datetime import datetime
# Save audio file
import soundfile as sf

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperManager:
    def __init__(self, threshold=0.03, input_device=None, output_device=None):
        try:
            # Load Whisper model
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logging.info("Using CUDA for Whisper model")

            # Audio settings
            self.threshold = threshold
            self.input_device = input_device
            self.output_device = output_device
            self.audio_queue = queue.Queue()
            self.buffer = []
            self.sample_rate = None  # Will be set when starting stream
            self.model_sample_rate = 16000  # Required by Whisper
            self.device_sample_rate = 44100.0
            self.stream = None

            # Speech detection settings
            self.is_buffering = False
            self.last_speech_time = None
            self.silence_duration = 0.8
            self.min_speech_duration = 0.2
            self.speech_start_time = None

            # Add audio file tracking
            self.last_audio_file = None
            self.audio_save_dir = Path("whisper_audio")
            self.audio_save_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            logging.error(f"Failed to initialize WhisperManager: {e}")
            raise

    def save_audio_segment(self, audio_array, sample_rate):
        """Save audio segment to WAV file and return the path"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"whisper_segment_{timestamp}.wav"
            filepath = self.audio_save_dir / filename
            
            # Ensure audio is float32 and normalized
            audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 0:
                audio_array = audio_array / np.abs(audio_array).max()
            
            sf.write(str(filepath), audio_array, sample_rate)
            
            return str(filepath)
            
        except Exception as e:
            logging.error(f"Error saving audio segment: {e}")
            return None

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            logging.warning(f"Sounddevice status: {status}")
        
        try:
            # Get the audio data from the output stream
            audio = outdata.copy()
            
            # Ensure we have the right shape
            if audio.ndim == 2:
                if audio.shape[1] > 1:  # If stereo
                    audio = np.mean(audio, axis=1)  # Convert to mono
                else:  # If mono in 2D array
                    audio = audio.flatten()
            
            # Ensure indata is not empty and has the expected shape
            if audio is None or audio.size == 0:
                logging.warning("Empty audio data received")
                return
            
            # Convert to mono if needed and ensure float32
            audio = audio.astype(np.float32)
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio**2))
            current_time = time.time()

            if rms > self.threshold:
                if not self.is_buffering:
                    logging.info(f"Speech detected! RMS: {rms:.6f}")
                    self.is_buffering = True
                    self.speech_start_time = current_time
                    # Add a small pre-buffer
                    pre_buffer = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                    self.buffer.extend(pre_buffer.tolist())
                self.buffer.extend(audio.tolist())
                self.last_speech_time = current_time
            else:
                if self.is_buffering:
                    self.buffer.extend(audio.tolist())
                    speech_duration = current_time - self.speech_start_time if self.speech_start_time else 0
                    silence_duration = current_time - self.last_speech_time if self.last_speech_time else 0
                    
                    if (speech_duration >= self.min_speech_duration and 
                        silence_duration >= self.silence_duration):
                        logging.info(f"Speech ended - Duration: {speech_duration:.2f}s")
                        self.is_buffering = False
                        
                        # Convert buffer to numpy array
                        full_audio = np.array(self.buffer, dtype=np.float32)
                        self.buffer = []  # Clear buffer
                        
                        if full_audio.size > 0:
                            # Save audio segment
                            audio_file = self.save_audio_segment(full_audio, self.sample_rate)
                            if audio_file:
                                self.last_audio_file = audio_file
                            
                            # Normalize audio
                            if np.abs(full_audio).max() > 0:
                                full_audio = full_audio / np.abs(full_audio).max()
                            
                            self.audio_queue.put({
                                "array": full_audio,
                                "sampling_rate": self.sample_rate
                            })
                            logging.info(f"Added audio segment to queue. Length: {len(full_audio)/self.sample_rate:.2f}s")
                        else:
                            logging.warning("Empty audio segment discarded")
                        
                        self.speech_start_time = None
                        
        except Exception as e:
            logging.error(f"Error in audio callback: {str(e)}", exc_info=True)

    def resample_audio(self, audio_array, orig_sample_rate):
        """Resample audio to match Whisper's expected sample rate"""
        if orig_sample_rate != self.model_sample_rate:
            # Calculate number of samples for target length
            target_length = int(len(audio_array) * self.model_sample_rate / orig_sample_rate)
            # Resample audio
            resampled_audio = scipy.signal.resample(audio_array, target_length)
            return resampled_audio
        return audio_array

    def transcribe_audio(self, audio_input):
        try:
            # Input validation
            if not isinstance(audio_input, dict):
                logging.error("Audio input must be a dictionary")
                return ""
            
            if 'array' not in audio_input or 'sampling_rate' not in audio_input:
                logging.error("Audio input must contain 'array' and 'sampling_rate'")
                return ""

            # Get the audio array
            audio_array = audio_input['array']
            
            # Convert to numpy array if needed
            if not isinstance(audio_array, np.ndarray):
                try:
                    audio_array = np.array(audio_array, dtype=np.float32)
                except Exception as e:
                    logging.error(f"Failed to convert audio to numpy array: {e}")
                    return ""
            
            # Validate audio array
            if audio_array.size == 0:
                logging.error("Audio array is empty")
                return ""
            
            if not np.isfinite(audio_array).all():
                logging.error("Audio array contains invalid values")
                return ""

            # Ensure audio is float32 and normalized
            audio_array = audio_array.astype(np.float32)
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val

            # Resample audio if needed
            if audio_input["sampling_rate"] != self.model_sample_rate:
                logging.info(f"Resampling audio from {audio_input['sampling_rate']}Hz to {self.model_sample_rate}Hz")
                audio_array = self.resample_audio(
                    audio_array, 
                    audio_input["sampling_rate"]
                )

            # Process with whisper
            input_features = self.processor(
                audio_array, 
                sampling_rate=self.model_sample_rate, 
                return_tensors="pt"
            ).input_features

            if torch.cuda.is_available():
                input_features = input_features.to("cuda")
            
            # Generate transcription
            with torch.no_grad():
                try:
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=448,
                        num_beams=5,
                        temperature=0.0
                    )
                    transcription = self.processor.batch_decode(
                        predicted_ids, 
                        skip_special_tokens=True
                    )
                    
                    if not transcription or not transcription[0].strip():
                        logging.warning("Empty transcription result")
                        return ""

                    result = transcription[0].strip()
                    logging.info(f"Successful transcription: {result}")
                    return result

                except Exception as e:
                    logging.error(f"Model generation failed: {e}")
                    return ""

        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            return ""

    def start_listening(self, sample_rate=None, channels=2):
        """Start the audio stream"""
        try:
            # Clean up any existing stream
            self.stop_listening()
            
            if self.input_device is None:
                raise ValueError("No input device specified")

            # Query device info for input
            device_info = sd.query_devices(self.input_device, 'input')
            logging.info(f"Full device info: {device_info}")
            
            # Get host API info
            host_api = sd.query_hostapis(device_info['hostapi'])
            logging.info(f"Using host API: {host_api['name']}")
            
            # Force sample rate to match device's native rate
            self.sample_rate = int(device_info['default_samplerate'])
            
            # Use the device's actual number of channels
            input_channels = min(device_info['max_input_channels'], 2)  # Use 1 or 2 channels
            logging.info(f"Using {input_channels} channel(s) for input")
            
            try:
                self.stream = sd.InputStream(
                    device=self.input_device,
                    channels=input_channels,
                    samplerate=self.sample_rate,
                    callback=self.audio_callback,
                    dtype=np.float32,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                    latency='low'
                )
                
                # Start the stream
                self.stream.start()
                logging.info(f"Started audio stream: device={self.input_device}, "
                            f"rate={self.sample_rate}Hz, channels={input_channels}")
                return True

            except Exception as e:
                logging.error(f"Failed to start audio stream: {e}")
                self.stop_listening()
                raise

        except Exception as e:
            logging.error(f"Failed to start audio stream: {e}")
            self.stop_listening()
            raise

    def stop_listening(self):
        """Stop and clean up the audio stream"""
        if self.stream is not None:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None

    def get_transcription(self):
        if not self.audio_queue.empty():
            audio_data = self.audio_queue.get()
            return self.transcribe_audio(audio_data)
        return "No speech detected."

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            self.stop_listening()
            # Optionally clean up old audio files
            # Uncomment if you want to delete old audio files
            # for file in self.audio_save_dir.glob("*.wav"):
            #     file.unlink()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
