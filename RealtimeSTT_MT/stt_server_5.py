"""
Advanced Speech-to-Text (STT) Server with Stable, Decoupled Translation

This server provides high-accuracy, real-time speech-to-text (STT)
transcription. It uses a robust producer-consumer pattern to handle
translation in a separate, non-blocking process, ensuring that the core
transcription performance and accuracy are not compromised.

This is the complete version with all command-line arguments and helper
callbacks restored.
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

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

from colorama import init, Fore, Style
init()

from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import numpy as np
import websockets

# --- GLOBALS ---
global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
recorder_thread = None
stop_recorder = False
main_loop = None

# VAD state
prev_text = ""
text_time_deque = deque()
wav_file = None

# Real-time processing state
last_realtime_text = ""

# Queues
data_connections = set()
audio_queue = asyncio.Queue()
translation_queue = asyncio.Queue()

# Flags
debug_logging = False
extended_logging = False
log_incoming_chunks = False
writechunks = False
silence_timing = False

# Configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1

def preprocess_text(text):
    """Cleans up transcription text."""
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:].lstrip()
    return text

def debug_print(message):
    """Prints a debug message if debug_logging is enabled."""
    if debug_logging:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

# ==============================================================================
# 1. TRANSLATION WORKER (Consumer)
# ==============================================================================

async def call_nllb_model(text_to_translate, target_language):
    """ Placeholder for the actual NLLB translation model call. """
    debug_print(f"Translating '{text_to_translate}' to {target_language}...")
    await asyncio.sleep(0.5) # Simulate network/model latency
    translated_text = f"({target_language}) {text_to_translate}"
    debug_print(f"Translation result: '{translated_text}'")
    return translated_text

async def translation_worker():
    """ Consumes text from translation_queue, translates it, and sends the result. """
    print(f"{bcolors.OKCYAN}Translation worker started.{bcolors.ENDC}")
    while True:
        job = await translation_queue.get()
        text = job['text']
        is_final = job['is_final']

        translated_text = await call_nllb_model(text, target_language=global_args.target_language)

        message = json.dumps({
            'type': 'translationUpdate',
            'text': translated_text,
            'is_final': is_final
        })
        await audio_queue.put(message)

# ==============================================================================
# 2. CALLBACK FUNCTIONS (Producers)
# ==============================================================================

def text_detected(text, loop):
    """ Callback for real-time transcription. Handles VAD timing and translation jobs. """
    global prev_text, last_realtime_text

    text = preprocess_text(text)

    # Dynamic VAD timing adjustment (restored from stt_server_2.py for accuracy)
    if silence_timing:
        def ends_with_ellipsis(s: str):
            return s.endswith("...") or (len(s) > 1 and s[:-1].endswith("..."))

        def sentence_end(s: str):
            return s and s[-1] in ['.', '!', '?', '。']

        if ends_with_ellipsis(text):
            recorder.post_speech_silence_duration = global_args.mid_sentence_detection_pause
        elif sentence_end(text) and sentence_end(prev_text) and not ends_with_ellipsis(prev_text):
            recorder.post_speech_silence_duration = global_args.end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause

    # Detect new words for low-latency translation
    new_text_segment = ""
    if text.startswith(last_realtime_text):
        new_text_segment = text[len(last_realtime_text):].strip()
    else:
        new_text_segment = text # It's a full revision
    
    if new_text_segment:
        translation_job = {'text': new_text_segment, 'is_final': False}
        asyncio.run_coroutine_threadsafe(translation_queue.put(translation_job), main_loop)

    last_realtime_text = text
    prev_text = text

    # Send real-time transcription to client
    message = json.dumps({'type': 'realtime', 'text': text})
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), main_loop)

    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')

def process_text(full_sentence):
    """ Callback for a finalized sentence. """
    global prev_text, last_realtime_text

    # Reset state for the next utterance
    prev_text = ""
    last_realtime_text = ""
    
    full_sentence = preprocess_text(full_sentence)

    # Queue final, high-quality translation job
    final_translation_job = {'text': full_sentence, 'is_final': True}
    asyncio.run_coroutine_threadsafe(translation_queue.put(final_translation_job), main_loop)

    # Send final transcription to client
    message = json.dumps({'type': 'fullSentence', 'text': full_sentence})
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), main_loop)
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"\n[{timestamp}] Original Sentence: {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}")

# ==============================================================================
# 3. SERVER BOILERPLATE AND SETUP
# ==============================================================================

def on_recording_start(loop):
    debug_print("Recording started")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'recording_start'})), loop)

def on_recording_stop(loop):
    debug_print("Recording stopped")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'recording_stop'})), loop)

def on_vad_detect_start(loop):
    debug_print("VAD picked up speech")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'vad_detect_start'})), loop)

def on_vad_detect_stop(loop):
    debug_print("VAD detected silence")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'vad_detect_stop'})), loop)

def on_turn_detection_start(loop):
    debug_print("Turn detection started")
    print("&&& stt_server on_turn_detection_start")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'start_turn_detection'})), loop)

def on_turn_detection_stop(loop):
    debug_print("Turn detection stopped")
    print("&&& stt_server on_turn_detection_stop")
    asyncio.run_coroutine_threadsafe(audio_queue.put(json.dumps({'type': 'stop_turn_detection'})), loop)

def parse_arguments():
    """Defines and parses the server's command-line arguments (Complete Version)."""
    global debug_logging, extended_logging, writechunks, log_incoming_chunks, silence_timing
    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server.')

    parser.add_argument('--target_language', type=str, default='eng_Latn', help='Target language for NLLB translation (e.g., eng_Latn, fra_Latn).')
    parser.add_argument('-m', '--model', type=str, default='large-v3', help='Path to the STT model or model size. Options: tiny, base, small, medium, large-v1, large-v2, large-v3, or a CTranslate2 model path. Default is large-v3.')
    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='small', help='Model size for real-time transcription. Same options as --model. Default is small.')
    parser.add_argument('-l', '--lang', '--language', type=str, default='', help="Language code for transcription (e.g., 'en', 'de'). Leave empty for auto-detection. Default is '' (auto-detect).")
    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012, help='Port for the data WebSocket connection. Default is 8012.')
    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1, help='Index of the audio input device. Default is 1.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model to use ("cuda" or "cpu"). Default is "cuda".')
    parser.add_argument('--gpu_device_index', type=int, default=0, help='Index of the GPU device to use. Default is 0.')
    parser.add_argument('-D', '--debug', action='store_true', help='Enable detailed debug logging.')
    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for websockets.')
    parser.add_argument('--use_extended_logging', action='store_true', help='Enable extended logging for the recording worker.')
    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file.')
    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunk arrivals.')
    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. Default is 16.')
    parser.add_argument('--compute_type', type=str, default='default', help='Type of computation for CTranslate2.')
    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True, help='Enable continuous real-time transcription.')
    parser.add_argument('--min_length_of_recording', type=float, default=1.1, help='Minimum duration of valid recordings in seconds.')
    parser.add_argument('--min_gap_between_recordings', type=float, default=0, help='Minimum gap in seconds between consecutive recordings.')
    parser.add_argument('--initial_prompt', type=str, default="Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'", help='Initial prompt to guide the transcription model.')
    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='List of token IDs to suppress during transcription.')
    parser.add_argument('--faster_whisper_vad_filter', action='store_true', help='Enable VAD filter for Faster Whisper.')
    parser.add_argument('--use_main_model_for_realtime', action='store_true', help='Use the main model for real-time transcription.')
    parser.add_argument('--realtime_processing_pause', type=float, default=0.02, help='Pause in seconds between processing real-time audio chunks.')
    parser.add_argument('--init_realtime_after_seconds', type=float, default=0.2, help='Initial delay before real-time transcription starts.')
    parser.add_argument('--realtime_batch_size', type=int, default=16, help='Batch size for the real-time model.')
    parser.add_argument('--initial_prompt_realtime', type=str, default="", help='Initial prompt for the real-time model.')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for the main transcription model.')
    parser.add_argument('--beam_size_realtime', type=int, default=5, help='Beam size for the real-time transcription model.')
    parser.add_argument('-s', '--silence_timing', action='store_true', default=True, help='Enable dynamic adjustment of silence duration.')
    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True, help='Use Silero model for end-of-speech detection.')
    parser.add_argument('--early_transcription_on_silence', type=float, default=0.2, help='Start transcription after this many seconds of silence mid-speech.')
    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.45, help='Silence duration to interpret as end of a sentence.')
    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.7, help='Silence duration to interpret as an incomplete sentence.')
    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0, help='Silence duration to interpret as a mid-sentence break.')
    parser.add_argument('--silero_sensitivity', type=float, default=0.1, help='Silero VAD sensitivity (0-1).')
    parser.add_argument('--silero_use_onnx', action='store_true', default=False, help='Use ONNX version of Silero model.')
    parser.add_argument('--webrtc_sensitivity', type=int, default=1, help='WebRTC VAD sensitivity (0-3).')
    parser.add_argument('--wake_words', type=str, default="", help='Comma-separated wake word(s).')
    
    args = parser.parse_args()
    debug_logging = args.debug
    extended_logging = args.use_extended_logging
    writechunks = args.write
    log_incoming_chunks = args.logchunks
    silence_timing = args.silence_timing
    return args

def _recorder_thread():
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
    global global_args
    data_connections.add(websocket)
    print(f"{bcolors.OKGREEN}Data client connected.{bcolors.ENDC}")
    
    try:
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

        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                recorder.feed_audio(message)
    except Exception as e:
        print(f"{bcolors.FAIL}Error in data_handler: {e}{bcolors.ENDC}")
    finally:
        data_connections.remove(websocket)

async def broadcast_audio_messages():
    """Broadcasts messages from the audio queue to all connected clients."""
    while True:
        message = await audio_queue.get()
        if data_connections:
            await asyncio.gather(*[conn.send(message) for conn in data_connections])

def make_callback(loop, callback_func):
    """Creates a thread-safe wrapper for callbacks."""
    def wrapper(*args, **kwargs):
        loop.call_soon_threadsafe(partial(callback_func, *args, **kwargs, loop=loop))
    return wrapper

async def main_async():
    """Main asynchronous function to set up and run the server."""
    global recorder_config, global_args, main_loop, recorder_thread, stop_recorder
    args = parse_arguments()
    global_args = args
    main_loop = asyncio.get_event_loop()

    recorder_config = {
        key: getattr(args, key) for key in dir(args) if not key.startswith('_')
    }
    recorder_config.update({
        'use_microphone': False,
        'spinner': False,
        'on_realtime_transcription_update': make_callback(main_loop, text_detected),
        'on_recording_start': make_callback(main_loop, on_recording_start),
        'on_recording_stop': make_callback(main_loop, on_recording_stop),
        'on_vad_detect_start': make_callback(main_loop, on_vad_detect_start),
        'on_vad_detect_stop': make_callback(main_loop, on_vad_detect_stop),
        'on_turn_detection_start': make_callback(main_loop, on_turn_detection_start),
        'on_turn_detection_stop': make_callback(main_loop, on_turn_detection_stop),
        'level': logging.DEBUG if args.debug else logging.WARNING
    })

    try:
        recorder_thread = threading.Thread(target=_recorder_thread, daemon=True)
        recorder_thread.start()
        recorder_ready.wait()

        data_server = await websockets.serve(data_handler, "0.0.0.0", args.data)
        print(f"{bcolors.OKGREEN}Data server listening on ws://0.0.0.0:{args.data}{bcolors.ENDC}")
        
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
        stop_recorder = True
        if recorder:
            recorder.shutdown()
        if recorder_thread:
            recorder_thread.join()
        print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")

def main():
    """Main entry point of the script."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Main function interrupted.{bcolors.ENDC}")
    sys.exit(0)

if __name__ == '__main__':
    main()
