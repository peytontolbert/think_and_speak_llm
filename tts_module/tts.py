import torch
import torchaudio
import soundfile as sf
import os
import tempfile
from types import SimpleNamespace
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    target_sample_rate,
)
import logging
import time
import sounddevice as sd
import numpy as np
from typing import Optional

class F5TTSService:
    def __init__(self, model_dir="G:/365daychallenge/projects/roadmap_projects/unfinished_projects/think_and_speak_llm", voice_profile="Bane"):
        """Initialize F5 TTS service with model and voice profile"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Validate model directory
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
        self.model_dir = model_dir
        self.voice_profile = voice_profile
        
        # Setup paths with logging
        self.checkpoint_path = os.path.join(model_dir, "weights", "final_finetuned_model.pt")
        self.vocab_path = os.path.join(model_dir, "F5TTS_Base_vocab.txt") 
        self.voice_profile_dir = os.path.join(model_dir, "voice_profiles", voice_profile)
        
        # Log paths for debugging
        logging.info(f"Model checkpoint path: {self.checkpoint_path}")
        logging.info(f"Vocab path: {self.vocab_path}")
        logging.info(f"Voice profile dir: {self.voice_profile_dir}")
        
        # Validate critical paths
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {self.vocab_path}")
        if not os.path.exists(self.voice_profile_dir):
            raise FileNotFoundError(f"Voice profile not found at: {self.voice_profile_dir}")
        
        # Initialize components
        self.vocab_char_map = None
        self.model = None
        self.vocoder = None
        self.ref_audio = None
        self.ref_text = None
        self.output_device = None
        
        # Load everything
        if not self.initialize():
            raise RuntimeError("Failed to initialize F5TTS Service")
        
    def initialize(self):
        """Initialize model, vocoder and reference audio"""
        try:
            # Load vocabulary
            self.vocab_char_map, vocab_size = self._load_vocab()
            
            # Initialize model
            self.model = self._create_model(vocab_size)
            
            # Load checkpoint
            self._load_checkpoint()
            
            # Load vocoder
            self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
            
            # Load reference audio
            self._load_reference_audio()
            
            logging.info(f"F5TTS Service initialized successfully using {self.device}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize F5TTS Service: {e}")
            return False

    def _load_vocab(self):
        """Load vocabulary from file"""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f.readlines()]
            
            vocab_char_map = {}
            for i, char in enumerate(vocab):
                if char:  # Skip empty lines
                    vocab_char_map[char] = i
                    
            return vocab_char_map, len(vocab_char_map) + 1
            
        except Exception as e:
            logging.error(f"Error loading vocabulary: {e}")
            raise

    def _create_model(self, vocab_size):
        """Create and configure the model"""
        try:
            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4
            )

            mel_spec_kwargs = dict(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=100,
                target_sample_rate=24000,
                mel_spec_type="vocos"
            )

            model = CFM(
                transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
                mel_spec_kwargs=mel_spec_kwargs,
                vocab_char_map=self.vocab_char_map,
            )
            
            return model.to(self.device)
            
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            raise

    def _load_checkpoint(self):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise

    def _load_reference_audio(self):
        """Load reference audio from voice profile"""
        try:
            samples_file = os.path.join(self.voice_profile_dir, "samples.txt")
            if not os.path.exists(samples_file):
                raise FileNotFoundError(f"Voice profile samples not found: {samples_file}")
                
            with open(samples_file, 'r') as f:
                first_sample = f.readline().strip().split('|')
                if len(first_sample) != 2:
                    raise ValueError("Invalid sample format in samples.txt")
                    
                audio_file, text = first_sample
                self.ref_audio, self.ref_text = preprocess_ref_audio_text(audio_file, text)
                
        except Exception as e:
            logging.error(f"Error loading reference audio: {e}")
            raise

    def synthesize(self, text, output_path=None):
        """
        Synthesize speech from text
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file. If None, creates temp file
        Returns:
            Path to the generated audio file
        """
        if not text or not isinstance(text, str):
            logging.error(f"Invalid text input: {text}")
            return None
        
        try:
            logging.info(f"Starting synthesis for text: {text[:50]}...")
            
            # Validate model state
            if self.model is None or self.vocoder is None or self.ref_audio is None or self.ref_text is None:
                logging.error("Model components not fully initialized")
                return None
            
            # Generate audio with torch.no_grad() for efficiency
            with torch.no_grad():
                # Generate audio
                logging.info("Generating audio...")
                audio, sample_rate, _ = infer_process(
                    self.ref_audio,
                    self.ref_text,
                    text,
                    self.model,
                    self.vocoder,
                    mel_spec_type="vocos",
                    speed=1.0,
                    nfe_step=32,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0
                )
                
                # Wait for GPU operations to complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                if audio is None or len(audio) == 0:
                    logging.error("Generated audio is empty or invalid")
                    return None
                
                # Setup output path
                if output_path is None:
                    temp_dir = os.path.join(self.voice_profile_dir, "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    output_path = os.path.join(temp_dir, f"speech_{int(time.time())}.wav")
                
                # Save audio file with high quality settings
                logging.info(f"Saving audio to {output_path}")
                try:
                    sf.write(
                        output_path,
                        audio,
                        sample_rate,
                        'PCM_16',
                        format='WAV'
                    )
                    # Ensure file is written completely
                    sf.SoundFile(output_path).close()
                except Exception as write_error:
                    logging.error(f"Failed to write audio file: {write_error}")
                    return None
                
                # Verify file
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    logging.error("Output file is invalid")
                    return None
                
                logging.info("Successfully generated speech file")
                return output_path
            
        except Exception as e:
            logging.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            return None

    def cleanup(self):
        """Cleanup any temporary files and resources"""
        try:
            temp_dir = os.path.join(self.voice_profile_dir, "temp")
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except Exception as e:
                        logging.warning(f"Failed to remove temp file {file}: {e}")
                        
        except Exception as e:
            logging.error(f"Error during cleanup: {e}") 

    def play_speech(self, output_path: str, output_device: Optional[int] = None) -> None:
        """
        Play synthesized speech through the designated voice device
        Args:
            output_path: Path to the synthesized audio file
            output_device: Optional output device ID
        """
        try:
            if not os.path.exists(output_path):
                logging.error(f"Audio file does not exist: {output_path}")
                return
            
            # Read the audio file
            data, samplerate = sf.read(output_path)
            
            # Convert to float32 if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Normalize audio to prevent clipping
            data = data / np.max(np.abs(data))
    
            # Play audio through the specified device
            try:
                sd.play(data, samplerate, device=output_device, blocking=True)
                sd.wait()  # Wait until audio is finished playing
            except Exception as e:
                logging.error(f"Error during playback: {e}")
                
        except Exception as e:
            logging.error(f"Error playing audio: {e}") 