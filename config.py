import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '150'))
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

# Elevenlabs Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_monolingual_v1')

# LLM Configuration
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.')
CHAT_HISTORY_SIZE = int(os.getenv('CHAT_HISTORY_SIZE', '1'))

# Whisper STT Configuration
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'distil-whisper/distil-large-v3')

# VAD Configuration
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', '0.3'))
VAD_MIN_SPEECH_DURATION_MS = int(os.getenv('VAD_MIN_SPEECH_DURATION_MS', '250'))
VAD_MAX_SPEECH_DURATION_MS = float(os.getenv('VAD_MAX_SPEECH_DURATION_MS', 'inf'))
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv('VAD_MIN_SILENCE_DURATION_MS', '100'))
VAD_SPEECH_PAD_MS = int(os.getenv('VAD_SPEECH_PAD_MS', '30'))

# Socket Configuration
SOCKET_HOST = os.getenv('SOCKET_HOST', 'localhost')
SOCKET_SEND_PORT = int(os.getenv('SOCKET_SEND_PORT', '12345'))
SOCKET_RECV_PORT = int(os.getenv('SOCKET_RECV_PORT', '12346'))
