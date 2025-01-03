import os
from src.conversation_manager import ConversationManager
import sounddevice as sd
import logging
from config import load_config
logger = logging.getLogger(__name__)

def get_default_devices():
    """Get default input and output devices"""
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        
        # Validate default input device
        input_info = devices[default_input]
        if input_info['max_input_channels'] == 0:
            # Find first valid input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default_input = i
                    input_info = device
                    break
        
        input_name = devices[default_input]['name']
        output_name = devices[default_output]['name']
        
        logger.info(f"Default input device: {input_name}")
        logger.info(f"Default output device: {output_name}")
        
        return default_input, default_output
    except Exception as e:
        logger.error(f"Error getting default devices: {e}")
        return None, None

def list_audio_devices():
    """List available audio devices"""
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")

    default_input, default_output = sd.default.device
    print(f"\nDefault input device: {devices[default_input]['name']}")
    print(f"Default output device: {devices[default_output]['name']}")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get default devices
    default_input, default_output = get_default_devices()
    if default_input is None or default_output is None:
        logger.error("Could not get default audio devices")
        return
    
    # List audio devices
    list_audio_devices()
    
    # Ask if user wants to use default devices
    use_default = input("\nUse default audio devices? (y/n): ").lower().strip() == 'y'
    
    if use_default:
        device_id = default_input
    else:
        device_id = int(input("Enter the number of your microphone device: "))
    
    # Get OpenAI API key
    try:
        config = load_config()
        api_key = config["openai_api_key"]
    except Exception as e:
        logger.error("Failed to load API key from .env file")
        raise
    
    # Create and start conversation manager
    manager = ConversationManager(
        api_key=api_key,
        input_device=device_id,
        output_device=default_output if use_default else None
    )
    
    try:
        manager.start()
        print("\nConversation system is running. Speak naturally. Press Ctrl+C to exit.")
        
        # Keep the main thread running
        while True:
            input()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        manager.stop()

if __name__ == "__main__":
    main() 