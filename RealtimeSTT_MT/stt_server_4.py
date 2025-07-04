"""
Advanced Speech-to-Text (STT) Server with Real-Time, Decoupled Translation

This server provides real-time speech-to-text (STT) transcription using the
RealtimeSTT library. It has been architected to support low-latency,
word-by-word processing for features like real-time translation.

It uses a producer-consumer pattern to decouple the translation task from
the transcription task, ensuring the STT engine remains responsive. The server
expects an initial configuration message from the client to set the target
translation language.

"""

import os
import sys
import subprocess
import importlib
import logging
import textwrap
from functools import partial

# Set Hugging Face endpoint mirror if needed
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def check_and_install_packages(packages):
    """
    Checks if specified Python packages are installed, and installs them if not.
    """
    for package in packages:
        module_name = package['module_name']
        attribute = package.get('attribute')
        install_name = package['install_name']

        try:
            if attribute:
                module = importlib.import_module(module_name)
                getattr(module, attribute)
            else:
                importlib.import_module(module_name)
        except (ImportError, AttributeError):
            print(f"Module '{module_name}' not found. Installing '{install_name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
                print(f"Package '{install_name}' installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"FATAL: Failed to install '{install_name}'. Please install it manually using 'pip install {install_name}'. Error: {e}", file=sys.stderr)
                sys.exit(1)

# Check and install required packages
check_and_install_packages([
    {'module_name': 'RealtimeSTT', 'attribute': 'AudioToTextRecorder', 'install_name': 'RealtimeSTT'},
    {'module_name': 'websockets', 'install_name': 'websockets'},
    {'module_name': 'numpy', 'install_name': 'numpy'},
    {'module_name': 'scipy.signal', 'attribute': 'resample', 'install_name': 'scipy'},
    {'module_name': 'pyaudio', 'install_name': 'pyaudio'},
    {'module_name': 'colorama', 'install_name': 'colorama'}
])


from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
import asyncio
import pyaudio
import base64
import threading
import wave
import json
import time

# Set asyncio policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Define ANSI color codes for terminal output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"{bcolors.BOLD}{bcolors.OKCYAN}Starting server, please wait...{bcolors.ENDC}")

# Initialize colorama for cross-platform colored output
from colorama import init, Fore, Style
init()

from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import numpy as np
import websockets

# --- MODIFIED: Global variables for advanced state and session management ---
global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
recorder_thread = None
stop_recorder = False
prev_text = ""
text_time_deque = deque()
wav_file = None
main_loop = None

# New data structure for storing the entire conversation history
session_history = []
current_session_index = -1

# New queue to send text from the transcription callbacks to the translation worker.
translation_queue = asyncio.Queue()

# The existing audio_queue now handles all messages to be sent to the client
audio_queue = asyncio.Queue()
# -------------------------------------------------------------------------

# Logging and debugging flags
debug_logging = False
extended_logging = False
log_incoming_chunks = False
writechunks = False

# Configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Silence detection parameters
silence_timing = False
hard_break_even_on_background_noise = 3.0
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_similarity = 0.99
hard_break_even_on_background_noise_min_chars = 15

# Connection set
data_connections = set()

def preprocess_text(text):
    """Cleans up transcription text."""
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:].lstrip()
    if text.endswith("...'."):
        text = text[:-3]
    elif text.endswith("...'"):
        text = text[:-2]
    if text:
        text = text[0].upper() + text[1:]
    return text

def debug_print(message):
    """Prints a debug message if debug_logging is enabled."""
    if debug_logging:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

# ==============================================================================
# 1. NEW TRANSLATION WORKER (Consumer)
# ==============================================================================

async def call_nllb_model(text_to_translate, target_language):
    """
    Placeholder for the actual NLLB translation model call.
    Simulates network/model latency.
    """
    debug_print(f"Translating '{text_to_translate}' to {target_language}...")
    await asyncio.sleep(0.5) # Simulate network/model latency
    translated_text = f"({target_language}) {text_to_translate}"
    debug_print(f"Translation result: '{translated_text}'")
    return translated_text

async def translation_worker():
    """
    Runs in a background task, consuming text from the translation_queue,
    translating it, and sending the result to the client.
    """
    print(f"{bcolors.OKCYAN}Translation worker started.{bcolors.ENDC}")
    while True:
        job = await translation_queue.get()

        text_to_translate = job['text']
        session_index = job['session_index']
        is_final_translation = job['is_final']

        translated_text = await call_nllb_model(text_to_translate, target_language=global_args.target_language)

        if len(session_history) > session_index:
            if is_final_translation:
                session_history[session_index]["final_translation"] = translated_text
            else:
                session_history[session_index]["translation_fragments"].append(translated_text)
            
        message = json.dumps({
            'type': 'translationUpdate',
            'text': translated_text,
            'session_index': session_index,
            'is_final': is_final_translation
        })
        await audio_queue.put(message)

# ==============================================================================
# 2. MODIFIED CALLBACK FUNCTIONS (Producers)
# ==============================================================================

def text_detected(text, loop):
    """
    Callback for real-time text. Now acts as a PRODUCER for the translation_queue.
    """
    global prev_text, session_history, current_session_index

    text = preprocess_text(text)
    
    if current_session_index < 0 or (len(session_history) > current_session_index and session_history[current_session_index]["is_final"]):
        current_session_index += 1
        session_history.append({
            "session_index": current_session_index,
            "fragments": [],
            "final_text": "",
            "is_final": False,
            "translation_fragments": [],
            "final_translation": ""
        })
        debug_print(f"New session created at index {current_session_index}")

    if len(session_history) > current_session_index:
        session_history[current_session_index]["fragments"].append(text)

    translation_job = {
        'text': text,
        'session_index': current_session_index,
        'is_final': False
    }
    asyncio.run_coroutine_threadsafe(translation_queue.put(translation_job), main_loop)
    
    message = json.dumps({'type': 'realtime', 'text': text})
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), main_loop)
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')

def process_text(full_sentence):
    """
    Callback for a finalized sentence. Locks the current session.
    """
    global session_history, current_session_index

    full_sentence = preprocess_text(full_sentence)
    
    if current_session_index >= 0 and len(session_history) > current_session_index:
        # Finalize and "lock" the current session
        session_history[current_session_index]["final_text"] = full_sentence
        session_history[current_session_index]["is_final"] = True
        debug_print(f"Session {current_session_index} finalized with text: '{full_sentence}'")
        
        # Queue one last translation job for the most accurate full sentence
        final_translation_job = {
            'text': full_sentence,
            'session_index': current_session_index,
            'is_final': True
        }
        asyncio.run_coroutine_threadsafe(translation_queue.put(final_translation_job), main_loop)

    # --- THE FIX: Add 'session_index' to the message ---
    message = json.dumps({
        'type': 'fullSentence',
        'text': full_sentence,
        'session_index': current_session_index # This key was missing
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), main_loop)
    
    # Standard console logging
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"\n[{timestamp}] Original Sentence: {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}")

# ==============================================================================
# 3. Restored Callbacks and Main Server Logic
# ==============================================================================

# Callback functions for various recorder events
def on_recording_start(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'recording_start'})), loop)

def on_recording_stop(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'recording_stop'})), loop)

def on_vad_detect_start(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'vad_detect_start'})), loop)

def on_vad_detect_stop(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'vad_detect_stop'})), loop)

def on_wakeword_detected(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'wakeword_detected'})), loop)

def on_wakeword_detection_start(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'wakeword_detection_start'})), loop)

def on_wakeword_detection_end(loop):
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'wakeword_detection_end'})), loop)

def on_transcription_start(_audio_bytes, loop):
    bytes_b64 = base64.b64encode(_audio_bytes).decode('utf-8')
    message = json.dumps({'type': 'transcription_start', 'audio_bytes_base64': bytes_b64})
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_start(loop):
    print("&&& stt_server on_turn_detection_start")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'start_turn_detection'})), loop)

def on_turn_detection_stop(loop):
    print("&&& stt_server on_turn_detection_stop")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'stop_turn_detection'})), loop)

def parse_arguments():
    """Defines and parses the server's command-line arguments (Complete Version)."""
    global debug_logging, extended_logging, writechunks, log_incoming_chunks, silence_timing
    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server.')

    # --- NEW: Argument for translation ---
    parser.add_argument('--target_language', type=str, default='eng_Latn', help='Target language for NLLB translation (e.g., eng_Latn, fra_Latn).')

    # Model and Language
    parser.add_argument('-m', '--model', type=str, default='large-v3', help='Path to the STT model or model size. Options: tiny, base, small, medium, large-v1, large-v2, large-v3, or a CTranslate2 model path. Default is large-v3.')
    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='large-v3', help='Model size for real-time transcription. Same options as --model. Default is small.')
    parser.add_argument('-l', '--lang', '--language', type=str, default='', help="Language code for transcription (e.g., 'en', 'de'). Leave empty for auto-detection. Default is '' (auto-detect).")

    # Server and Device
    parser.add_argument('-d', '--data', '--data_port', type=int, default=8013, help='Port for the data WebSocket connection. Default is 8012.')
    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1, help='Index of the audio input device. Default is 1.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model to use ("cuda" or "cpu"). Default is "cuda".')
    parser.add_argument('--gpu_device_index', type=int, default=0, help='Index of the GPU device to use. Default is 0.')

    # Debugging and Logging
    parser.add_argument('-D', '--debug', action='store_true', help='Enable detailed debug logging.')
    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for websockets.')
    parser.add_argument('--use_extended_logging', action='store_true', help='Enable extended logging for the recording worker.')
    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file.')
    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunk arrivals.')
    
    # Transcription and VAD
    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. Default is 16.')
    parser.add_argument('--compute_type', type=str, default='default', help='Type of computation for CTranslate2. See https://opennmt.net/CTranslate2/quantization.html.')
    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True, help='Enable continuous real-time transcription. Default is True.')
    parser.add_argument('--min_length_of_recording', type=float, default=1.1, help='Minimum duration of valid recordings in seconds. Default is 1.1.')
    parser.add_argument('--min_gap_between_recordings', type=float, default=0, help='Minimum gap in seconds between consecutive recordings. Default is 0.')
    parser.add_argument('--initial_prompt', type=str, default="Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'", help='Initial prompt to guide the transcription model.')
    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='List of token IDs to suppress during transcription. Default is [-1].')
    parser.add_argument('--faster_whisper_vad_filter', action='store_true', help='Enable VAD filter for Faster Whisper. Default is False.')

    # Real-time specific
    parser.add_argument('--use_main_model_for_realtime', action='store_true', help='Use the main model for real-time transcription instead of the dedicated rt-model.')
    parser.add_argument('--realtime_processing_pause', type=float, default=0.02, help='Pause in seconds between processing real-time audio chunks. Default is 0.02.')
    parser.add_argument('--init_realtime_after_seconds', type=float, default=0.2, help='Initial delay before real-time transcription starts. Default is 0.2.')
    parser.add_argument('--realtime_batch_size', type=int, default=16, help='Batch size for the real-time model. Default is 16.')
    parser.add_argument('--initial_prompt_realtime', type=str, default="", help='Initial prompt for the real-time model.')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for the main transcription model. Default is 5.')
    parser.add_argument('--beam_size_realtime', type=int, default=5, help='Beam size for the real-time transcription model. Default is 3.')
    
    # Silence Timing and VAD
    parser.add_argument('-s', '--silence_timing', action='store_true', default=True, help='Enable dynamic adjustment of silence duration. Default is True.')
    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True, help='Use Silero model for end-of-speech detection. Default is True.')
    parser.add_argument('--early_transcription_on_silence', type=float, default=0.1, help='Start transcription after this many seconds of silence mid-speech. Set to 0 to disable. Default is 0.2.')
    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.2, help='Silence duration to interpret as end of a sentence. Default is 0.45s.')
    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.5, help='Silence duration to interpret as an incomplete sentence. Default is 0.7s.')
    parser.add_argument('--mid_sentence_detection_pause', type=float, default=1.0, help='Silence duration to interpret as a mid-sentence break. Default is 2.0s.')
    parser.add_argument('--silero_sensitivity', type=float, default=0.1, help='Silero VAD sensitivity (0-1). Lower is less sensitive. Default is 0.05.')
    parser.add_argument('--silero_use_onnx', action='store_true', default=False, help='Use ONNX version of Silero model. Default is False.')
    parser.add_argument('--webrtc_sensitivity', type=int, default=1, help='WebRTC VAD sensitivity (0-3). Higher is less sensitive. Default is 3.')
    
    # Wake Word
    parser.add_argument('-w', '--wake_words', type=str, default="", help='Comma-separated wake word(s) to trigger listening. Default is "" (disabled).')
    parser.add_argument('--wakeword_backend', type=str, default='none', help='Wake word detection backend. Default is "none".')
    parser.add_argument('--wake_words_sensitivity', type=float, default=0.5, help='Wake word detection sensitivity (0-1). Default is 0.5.')
    parser.add_argument('--wake_word_timeout', type=float, default=5.0, help='Time in seconds to wait for a wake word. Default is 5.0.')
    parser.add_argument('--wake_word_activation_delay', type=float, default=0, help='Delay in seconds before wake word detection is active. Default is 0.')
    parser.add_argument('--wake_word_buffer_duration', type=float, default=1.0, help='Audio buffer duration in seconds for wake word detection. Default is 1.0.')
    parser.add_argument('--openwakeword_model_paths', type=str, nargs='*', help='File paths to OpenWakeWord models.')
    parser.add_argument('--openwakeword_inference_framework', type=str, default='tensorflow', help='Inference framework for OpenWakeWord. Default is "tensorflow".')
    
    # Advanced
    parser.add_argument('--root', '--download_root', type=str,default=None, help='Specifies the root path where the Whisper models are downloaded to. Default is None.')
    parser.add_argument('--handle_buffer_overflow', action='store_true', help='Handle buffer overflow during transcription. Default is False.')
    parser.add_argument('--allowed_latency_limit', type=int, default=500, help='Max unprocessed chunks in queue before discarding. Default is 100.')
    
    args = parser.parse_args()

    debug_logging = args.debug
    extended_logging = args.use_extended_logging
    writechunks = args.write
    log_incoming_chunks = args.logchunks
    silence_timing = args.silence_timing
    
    logging.getLogger('websockets').setLevel(logging.DEBUG if args.debug_websockets else logging.WARNING)

    if args.initial_prompt:
        args.initial_prompt = args.initial_prompt.replace("\\n", "\n")
    if args.initial_prompt_realtime:
        args.initial_prompt_realtime = args.initial_prompt_realtime.replace("\\n", "\n")

    return args

def _recorder_thread(loop):
    """Thread that runs the blocking AudioToTextRecorder."""
    global recorder, stop_recorder
    print(f"{bcolors.OKGREEN}Initializing RealtimeSTT with parameters:{bcolors.ENDC}")
    for key, value in recorder_config.items():
        if value is not None:
             print(f"    {bcolors.OKBLUE}{key}{bcolors.ENDC}: {value}")
    
    recorder = AudioToTextRecorder(**recorder_config)
    print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}RealtimeSTT initialized successfully.{bcolors.ENDC}")
    recorder_ready.set()

    try:
        while not stop_recorder:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Recorder thread interrupted.{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}Exception in recorder thread: {e}{bcolors.ENDC}")
    finally:
        stop_recorder = True

async def data_handler(websocket):
    """Handles WebSocket connections for audio data and initial config."""
    global global_args, session_history, current_session_index
    print(f"{bcolors.OKGREEN}Data client connected.{bcolors.ENDC}")
    data_connections.add(websocket)
    
    # --- NEW: Reset session state for a new connection ---
    session_history = []
    current_session_index = -1

    try:
        # --- NEW: Expect a configuration message first ---
        config_message = await websocket.recv()
        if isinstance(config_message, str):
            config = json.loads(config_message)
            if config.get('type') == 'config':
                lang = config.get('target_language', global_args.target_language)
                global_args.target_language = lang
                debug_print(f"Client set target language to: {lang}")
            else:
                raise ValueError("First message must be a config message.")
        else:
            raise ValueError("First message must be a JSON string for config.")

        # Now, enter the loop to process audio
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                chunk = message[4+metadata_length:]
                recorder.feed_audio(chunk)
            else:
                debug_print(f"Received non-binary message: {message}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Data client disconnected: {e}{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}Error in data_handler: {e}{bcolors.ENDC}")
    finally:
        data_connections.remove(websocket)
        if recorder:
            recorder.clear_audio_queue()

async def broadcast_audio_messages():
    """Broadcasts messages from the audio queue to all connected clients."""
    while True:
        message = await audio_queue.get()
        if data_connections:
            await asyncio.gather(*[conn.send(message) for conn in data_connections])

def make_callback(loop, callback_func):
    """Creates a thread-safe wrapper for callbacks."""
    def wrapper(*args, **kwargs):
        bound_callback = partial(callback_func, *args, **kwargs, loop=loop)
        loop.call_soon_threadsafe(bound_callback)
    return wrapper

async def main_async():
    """Main asynchronous function to set up and run the server."""
    global stop_recorder, recorder_config, global_args, recorder_thread, main_loop
    args = parse_arguments()
    global_args = args

    loop = asyncio.get_event_loop()
    main_loop = loop

    recorder_config = {
        'model': args.model, 'download_root': args.root, 'realtime_model_type': args.rt_model,
        'language': args.lang, 'batch_size': args.batch, 'init_realtime_after_seconds': args.init_realtime_after_seconds,
        'realtime_batch_size': args.realtime_batch_size, 'initial_prompt_realtime': args.initial_prompt_realtime,
        'input_device_index': args.input_device, 'silero_sensitivity': args.silero_sensitivity, 'silero_use_onnx': args.silero_use_onnx,
        'webrtc_sensitivity': args.webrtc_sensitivity, 'post_speech_silence_duration': args.unknown_sentence_detection_pause,
        'min_length_of_recording': args.min_length_of_recording, 'min_gap_between_recordings': args.min_gap_between_recordings,
        'enable_realtime_transcription': args.enable_realtime_transcription, 'realtime_processing_pause': args.realtime_processing_pause,
        'silero_deactivity_detection': args.silero_deactivity_detection, 'early_transcription_on_silence': args.early_transcription_on_silence,
        'beam_size': args.beam_size, 'beam_size_realtime': args.beam_size_realtime, 'initial_prompt': args.initial_prompt,
        'wake_words': args.wake_words, 'wake_words_sensitivity': args.wake_words_sensitivity, 'wake_word_timeout': args.wake_word_timeout,
        'wake_word_activation_delay': args.wake_word_activation_delay, 'wakeword_backend': args.wakeword_backend,
        'openwakeword_model_paths': args.openwakeword_model_paths, 'openwakeword_inference_framework': args.openwakeword_inference_framework,
        'wake_word_buffer_duration': args.wake_word_buffer_duration, 'use_main_model_for_realtime': args.use_main_model_for_realtime,
        'spinner': False, 'use_microphone': False, 'on_realtime_transcription_update': make_callback(loop, text_detected),
        'on_recording_start': make_callback(loop, on_recording_start), 'on_recording_stop': make_callback(loop, on_recording_stop),
        'on_vad_detect_start': make_callback(loop, on_vad_detect_start), 'on_vad_detect_stop': make_callback(loop, on_vad_detect_stop),
        'on_wakeword_detected': make_callback(loop, on_wakeword_detected), 'on_wakeword_detection_start': make_callback(loop, on_wakeword_detection_start),
        'on_wakeword_detection_end': make_callback(loop, on_wakeword_detection_end), 'on_transcription_start': make_callback(loop, on_transcription_start),
        'on_turn_detection_start': make_callback(loop, on_turn_detection_start), 'on_turn_detection_stop': make_callback(loop, on_turn_detection_stop),
        'no_log_file': True, 'use_extended_logging': args.use_extended_logging,
        'level': logging.DEBUG if args.debug else logging.WARNING, 'compute_type': args.compute_type, 'gpu_device_index': args.gpu_device_index,
        'device': args.device, 'handle_buffer_overflow': args.handle_buffer_overflow, 'suppress_tokens': args.suppress_tokens,
        'allowed_latency_limit': args.allowed_latency_limit, 'faster_whisper_vad_filter': args.faster_whisper_vad_filter,
    }

    try:
        recorder_thread = threading.Thread(target=_recorder_thread, args=(loop,), daemon=True)
        recorder_thread.start()
        recorder_ready.wait()

        data_server = await websockets.serve(data_handler, "0.0.0.0", args.data)
        print(f"{bcolors.OKGREEN}Data server listening on {bcolors.OKBLUE}ws://0.0.0.0:{args.data}{bcolors.ENDC}")
        
        broadcast_task = asyncio.create_task(broadcast_audio_messages())
        translation_task = asyncio.create_task(translation_worker())

        print(f"{bcolors.OKGREEN}Server started. Press Ctrl+C to stop.{bcolors.ENDC}")
        await asyncio.gather(
            data_server.wait_closed(),
            broadcast_task,
            translation_task
        )

    except OSError as e:
        print(f"{bcolors.FAIL}Error: Could not start server on port {args.data}. It may already be in use. ({e}){bcolors.ENDC}")
    except KeyboardInterrupt:
        print(f"\n{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
    finally:
        await shutdown_procedure()
        print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")

async def shutdown_procedure():
    """Gracefully shuts down the server and its components."""
    # ... (Shutdown logic remains the same)
    global stop_recorder, recorder_thread
    stop_recorder = True
    if recorder:
        recorder.shutdown()
    if recorder_thread and recorder_thread.is_alive():
        recorder_thread.join()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if wav_file:
        wav_file.close()

def main():
    """Main entry point of the script."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Main function interrupted.{bcolors.ENDC}")
    sys.exit(0)

if __name__ == '__main__':
    main()