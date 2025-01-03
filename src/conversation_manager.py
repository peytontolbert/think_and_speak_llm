import threading
import queue
import time
from typing import Optional, List, Dict
from src.whisper.whisper_manager import WhisperManager
from src.llm_interface.chatgpt import TextManager
from src.prompts.prompt_manager import PromptManager
from src.vector.vector_store import VectorStore
from tts_module.tts import F5TTSService
import logging
import sounddevice as sd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, api_key: str, input_device: Optional[int] = None, output_device: Optional[int] = None):
        self.text_manager = TextManager(api_key)
        self.vector_store = VectorStore()
        self.prompt_manager = PromptManager(self.text_manager, self.vector_store)
        self.output_device = output_device  # Store output device
        
        # Initialize TTS service
        try:
            self.tts_service = F5TTSService()
            logger.info("TTS Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS Service: {e}")
            raise
        
        # Validate input device
        if input_device is not None:
            device_info = sd.query_devices(input_device, 'input')
            if device_info['max_input_channels'] == 0:
                raise ValueError(f"Device {input_device} has no input channels")
        
        self.whisper_manager = WhisperManager(
            input_device=input_device,
            output_device=output_device
        )
        self.transcription_queue = queue.Queue()
        self.thought_queue = queue.Queue()
        self.should_run = True
        self.is_user_speaking = False
        self.speech_start_time = 0
        self.last_thought_time = 0
        self.is_ai_speaking = False
        self.current_utterance: List[str] = []
        self.speaking_duration = 0
        self.processing_lock = threading.Lock()
        self.last_cognitive_time = time.time()
        self.cognitive_interval = 2.0
        self.last_response_time = time.time()
        self.preprocessing_done = False
        self.response_cooldown = 3.0  # Minimum time between responses
        self.preprocessing_interval = 5.0  # Time after speaking to preprocess
        self.processed_messages = set()  # Track processed message IDs
        self.message_counter = 0  # For generating unique message IDs

    def _speak(self, text: str) -> None:
        """Generate and play speech for the given text"""
        try:
            # Generate speech
            output_path = self.tts_service.synthesize(text)
            if output_path:
                logger.info(f"Generated speech file: {output_path}")
                # Play the generated speech with the specified output device
                self.tts_service.play_speech(output_path, output_device=self.output_device)
            else:
                logger.error("Failed to generate speech")
        except Exception as e:
            logger.error(f"Error in speech generation/playback: {e}")

    def _handle_speech_detection(self, transcription: str) -> None:
        """Handle speech detection and update speaking states"""
        if transcription and transcription != "No speech detected.":
            if not self.is_user_speaking:
                self.is_user_speaking = True
                self.speech_start_time = time.time()
            self.current_utterance.append(transcription)
            self.speaking_duration = time.time() - self.speech_start_time
        else:
            if self.is_user_speaking:
                self.is_user_speaking = False
                self.speaking_duration = 0
                
    def _should_generate_thought(self) -> bool:
        """Determine if we should generate a new thought"""
        current_time = time.time()
        thought_interval = current_time - self.last_thought_time
        return (
            self.speaking_duration > 8 and  # User speaking for more than 8 seconds
            thought_interval > 5 and        # At least 5 seconds since last thought
            self.is_user_speaking           # User is currently speaking
        )
        
    def start(self):
        """Start the conversation system"""
        try:
            # Start audio recording
            self.whisper_manager.start_listening()
            
            # Start background threads
            threading.Thread(target=self._transcription_loop, daemon=True).start()
            threading.Thread(target=self._cognitive_loop, daemon=True).start()
            threading.Thread(target=self._conversation_loop, daemon=True).start()
            
            logger.info("Conversation system started")
            
        except Exception as e:
            logger.error(f"Failed to start conversation system: {e}")
            self.stop()
            
    def stop(self):
        """Stop the conversation system and cleanup"""
        self.should_run = False
        self.whisper_manager.cleanup()
        if hasattr(self, 'tts_service'):
            self.tts_service.cleanup()
        
    def _get_next_message_id(self) -> str:
        """Generate a unique message ID"""
        self.message_counter += 1
        return f"msg_{int(time.time())}_{self.message_counter}"

    def _transcription_loop(self):
        """Continuously process audio transcriptions"""
        while self.should_run:
            try:
                transcription = self.whisper_manager.get_transcription()
                
                if transcription and transcription != "No speech detected.":
                    message_id = self._get_next_message_id()
                    
                    # Update speaking state
                    if not self.is_user_speaking:
                        self.is_user_speaking = True
                        self.speech_start_time = time.time()
                        self.preprocessing_done = False
                    
                    # Put message in queue with ID
                    self.transcription_queue.put({
                        'id': message_id,
                        'text': transcription,
                        'timestamp': time.time()
                    })
                    
                    self.speaking_duration = time.time() - self.speech_start_time
                    
                    # If AI is speaking and user interrupts
                    if self.is_ai_speaking:
                        self.vector_store.add_text("[User interrupted AI's speech]")
                        self.is_ai_speaking = False
                else:
                    if self.is_user_speaking:
                        self.is_user_speaking = False
                        self.speaking_duration = 0
                
            except Exception as e:
                logger.error(f"Transcription error: {e}")
            time.sleep(0.1)

    def _cognitive_loop(self):
        """Enhanced background cognitive processing"""
        while self.should_run:
            try:
                current_time = time.time()
                # Only process if:
                # 1. Enough time has passed since last cognitive process
                # 2. Not currently speaking (AI or user)
                # 3. Not currently processing a response
                should_process = (
                    current_time - self.last_cognitive_time >= self.cognitive_interval and
                    not self.is_ai_speaking and
                    not self.is_user_speaking and
                    not getattr(self, 'processing_response', False)
                )
                
                if should_process:
                    with self.processing_lock:
                        # Process any interim thoughts
                        try:
                            thought = self.thought_queue.get_nowait()
                            if thought['type'] == 'interim_thought':
                                interim_response = self.prompt_manager.process_interim_thought(thought['context'])
                                if interim_response:  # Only add non-empty thoughts
                                    self.vector_store.add_text(f"[AI Thought: {interim_response}]")
                        except queue.Empty:
                            pass
                        
                        # Only do regular cognitive processing if we have context
                        context = self.vector_store.get_recent_context()
                        if context:  # Only process if we have context
                            cognitive_result = self.prompt_manager.cognitive_process(context)
                            
                            # Store non-speech cognitive results if they're meaningful
                            if cognitive_result and "<speak>" not in cognitive_result:
                                self.vector_store.add_text(f"[Internal Process: {cognitive_result}]")
                    
                    self.last_cognitive_time = current_time
                    
            except Exception as e:
                logger.error(f"Cognitive processing error: {e}")
                logger.exception("Full traceback:")
            
            # Sleep longer when speaking is happening
            sleep_time = 2.0 if self.is_ai_speaking or self.is_user_speaking else 1.0
            time.sleep(sleep_time)
    
    def _preprocess_context(self) -> None:
        """Preprocess context without generating speech"""
        try:
            context = self.vector_store.get_recent_context()
            cognitive_result = self.prompt_manager.cognitive_process(context)
            
            # Only store preprocessing results if they don't contain speech tags
            if "<speak>" not in cognitive_result:
                self.vector_store.add_text(f"[Preprocessing: {cognitive_result}]")
                self.preprocessing_done = True
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")

    def _conversation_loop(self):
        """Enhanced conversation flow with natural timing"""
        last_transcription_time = time.time()
        processing_response = False
        
        while self.should_run:
            try:
                with self.processing_lock:
                    current_time = time.time()
                    
                    # Check for new transcriptions
                    try:
                        message = self.transcription_queue.get_nowait()
                        message_id = message['id']
                        
                        # Only process if we haven't seen this message before
                        if message_id not in self.processed_messages:
                            transcription = message['text']
                            last_transcription_time = current_time
                            logger.info(f"Processing transcription: {transcription}")
                            
                            # Add to utterance and mark as processed
                            self.current_utterance.append(transcription)
                            self.processed_messages.add(message_id)
                            
                            # Limit size of processed messages set
                            if len(self.processed_messages) > 1000:
                                self.processed_messages = set(list(self.processed_messages)[-500:])
                            
                    except queue.Empty:
                        pass

                    silence_duration = current_time - last_transcription_time
                    response_delay = current_time - self.last_response_time
                    
                    # Preprocess after AI has finished speaking and before next response
                    if (not self.is_user_speaking and 
                        not self.is_ai_speaking and 
                        not self.preprocessing_done and
                        silence_duration > self.preprocessing_interval):
                        self._preprocess_context()
                    
                    # Only process response if:
                    # 1. We have utterances to process
                    # 2. User has stopped speaking or been speaking too long
                    # 3. Not currently processing/speaking
                    # 4. Enough time has passed since last response
                    should_process = (
                        not processing_response and
                        not self.is_ai_speaking and
                        self.current_utterance and
                        response_delay > self.response_cooldown and
                        (
                            (not self.is_user_speaking and silence_duration > 1.5) or
                            (self.is_user_speaking and self.speaking_duration > 8)
                        )
                    )
                    
                    if should_process:
                        processing_response = True
                        full_utterance = " ".join(self.current_utterance)
                        logger.info(f"Processing complete utterance: {full_utterance}")
                        
                        response = self.prompt_manager.conversational_response(full_utterance)
                        logger.info(f"Got AI response: {response}")
                        
                        if "<speak>" in response:
                            speech = response.split("<speak>")[1].split("</speak>")[0]
                            print(f"\nAI: {speech}")
                            self.is_ai_speaking = True
                            
                            # Generate and play speech
                            self._speak(speech)
                            self.last_response_time = current_time
                            
                            if "<uninterruptible>" in response:
                                time.sleep(len(speech.split()) * 0.3)
                            
                            self.is_ai_speaking = False
                        
                        self.current_utterance = []
                        processing_response = False
                        self.preprocessing_done = False  # Reset for next round
                    
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                logger.exception("Full traceback:")
                processing_response = False
            
            time.sleep(0.2)