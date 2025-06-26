"""
Simplified Speech-to-Text (STT) Server with Real-Time Transcription

This server provides real-time speech-to-text (STT) transcription using the
RealtimeSTT library. It allows clients to connect via a single WebSocket
to send audio data and receive real-time transcription updates.

This is a simplified version of the original server, with the control
WebSocket removed. All configuration is handled via command-line arguments
at startup.

### Starting the Server:
You can start the server by running this Python script directly,
passing the desired configuration options.

Example:
python your_script_name.py --model small --lang en --data 8012

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

    Args:
        packages (list of dict): A list of packages to check. Each dictionary
                                 should contain 'module_name', 'install_name',
                                 and optionally 'attribute'.
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

# Global variables
global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
recorder_thread = None
stop_recorder = False
prev_text = ""
text_time_deque = deque()
wav_file = None

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

# Queues and connections
data_connections = set()
audio_queue = asyncio.Queue()

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

def format_timestamp_ns(timestamp_ns: int) -> str:
    """Formats a nanosecond timestamp to HH:MM:SS.ms."""
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    time_str = dt.strftime("%H:%M:%S")
    milliseconds = remainder_ns // 1_000_000
    return f"{time_str}.{milliseconds:03d}"

def text_detected(text, loop):
    """Callback function for when real-time text is detected."""
    global prev_text
    text = preprocess_text(text)

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

        current_time = time.time()
        text_time_deque.append((current_time, text))

        while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
            text_time_deque.popleft()

        if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
            texts = [t[1] for t in text_time_deque]
            first_text = texts[0]
            last_text = texts[-1]
            similarity = SequenceMatcher(None, first_text, last_text).ratio()

            if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                recorder.stop()
                recorder.clear_audio_queue()
                prev_text = ""

    prev_text = text
    message = json.dumps({'type': 'realtime', 'text': text})
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    if extended_logging:
        print(f"  [{timestamp}] Realtime text: {bcolors.OKCYAN}{text}{bcolors.ENDC}\n", flush=True, end="")
    else:
        print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')

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
    """Defines and parses the server's command-line arguments."""
    global debug_logging, extended_logging, writechunks, log_incoming_chunks, silence_timing
    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server.')

    # Model and Language
    parser.add_argument('-m', '--model', type=str, default='large-v3', help='Path to the STT model or model size. Options: tiny, base, small, medium, large-v1, large-v2, large-v3, or a CTranslate2 model path. Default is large-v3.')
    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='small', help='Model size for real-time transcription. Same options as --model. Default is small.')
    parser.add_argument('-l', '--lang', '--language', type=str, default='', help="Language code for transcription (e.g., 'en', 'de'). Leave empty for auto-detection. Default is '' (auto-detect).")

    # Server and Device
    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012, help='Port for the data WebSocket connection. Default is 8012.')
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
    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.4, help='Silence duration to interpret as end of a sentence. Default is 0.45s.')
    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.7, help='Silence duration to interpret as an incomplete sentence. Default is 0.7s.')
    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0, help='Silence duration to interpret as a mid-sentence break. Default is 2.0s.')
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

    def process_text(full_sentence):
        """
        Callback for when a full sentence is transcribed.
        This function now acts as a processing pipeline.
        """
        global prev_text
        prev_text = ""
        print("previous text:", prev_text)
        full_sentence = preprocess_text(full_sentence)
        print("current text:", full_sentence)
        # --- NEW PROCESSING LOGIC ---
        MAX_LENGTH = 30  # Define our rule
        
        # Use textwrap to split the sentence into lines of max 80 chars
        # This is safer than manual splitting.
        processed_sentences = textwrap.wrap(full_sentence, width=MAX_LENGTH)

        # Now, instead of sending one message, we send a message for each processed part.
        for sentence_part in processed_sentences:
            message = json.dumps({'type': 'fullSentence', 'text': sentence_part})
            
            # The queuing mechanism remains the same
            # This will send each part as a separate message to the client
            asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

        # --- The original logging can be adjusted or kept as is ---
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        # Log the original, unprocessed sentence
        print(f"\r[{timestamp}] Original Sentence: {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}")
        # Log what was actually sent
        if len(processed_sentences) > 1:
            print(f"  ↳ Split into {len(processed_sentences)} parts for the client.")

    try:
        while not stop_recorder:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Recorder thread interrupted.{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}Exception in recorder thread: {e}{bcolors.ENDC}")
    finally:
        stop_recorder = True

def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
    """Resamples audio data to the target sample rate."""
    if original_sample_rate == target_sample_rate:
        return audio_data
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
    resampled_audio = resample(audio_np, num_target_samples)
    return resampled_audio.astype(np.int16).tobytes()

async def data_handler(websocket):
    """Handles WebSocket connections for audio data."""
    global writechunks, wav_file
    print(f"{bcolors.OKGREEN}Data client connected.{bcolors.ENDC}")
    data_connections.add(websocket)
    
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                if extended_logging:
                    debug_print(f"Received audio chunk (size: {len(message)} bytes)")
                elif log_incoming_chunks:
                    print(".", end='', flush=True)
                
                # Extract metadata and audio chunk
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']
                chunk = message[4+metadata_length:]

                if writechunks and chunk:
                    if not wav_file:
                        wav_file = wave.open(writechunks, 'wb')
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wav_file.setframerate(sample_rate)
                    wav_file.writeframes(chunk)

                if sample_rate != 16000:
                    chunk = decode_and_resample(chunk, sample_rate, 16000)
                
                recorder.feed_audio(chunk)
            else:
                print(f"{bcolors.WARNING}Received non-binary message on data connection.{bcolors.ENDC}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Data client disconnected: {e}{bcolors.ENDC}")
    finally:
        data_connections.remove(websocket)
        recorder.clear_audio_queue()

async def broadcast_audio_messages():
    """Broadcasts messages from the audio queue to all connected clients."""
    while True:
        message = await audio_queue.get()
        if data_connections:
            await asyncio.wait([conn.send(message) for conn in data_connections])

def make_callback(loop, callback_func):
    """
    Creates a wrapper that allows a function to be safely called from a
    background thread by scheduling it on the main asyncio event loop.
    """
    def wrapper(*args, **kwargs):
        # Create a function with all arguments pre-filled
        bound_callback = partial(callback_func, *args, **kwargs, loop=loop)
        # Schedule this pre-filled function to run on the loop
        loop.call_soon_threadsafe(bound_callback)
    return wrapper

async def main_async():
    """Main asynchronous function to set up and run the server."""
    global stop_recorder, recorder_config, global_args, recorder_thread
    args = parse_arguments()
    global_args = args

    loop = asyncio.get_event_loop()

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
        recorder_ready.wait() # Wait for recorder to be initialized

        data_server = await websockets.serve(data_handler, "0.0.0.0", args.data)
        print(f"{bcolors.OKGREEN}Data server listening on {bcolors.OKBLUE}ws://0.0.0.0:{args.data}{bcolors.ENDC}")
        
        broadcast_task = asyncio.create_task(broadcast_audio_messages())

        print(f"{bcolors.OKGREEN}Server started. Press Ctrl+C to stop.{bcolors.ENDC}")
        await asyncio.gather(data_server.wait_closed(), broadcast_task)

    except OSError as e:
        print(f"{bcolors.FAIL}Error: Could not start server on port {args.data}. It may already be in use. ({e}){bcolors.ENDC}")
    except KeyboardInterrupt:
        print(f"\n{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
    finally:
        await shutdown_procedure()
        print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")

async def shutdown_procedure():
    """Gracefully shuts down the server and its components."""
    global stop_recorder, recorder_thread
    stop_recorder = True
    
    if recorder:
        recorder.shutdown()
        print(f"{bcolors.OKGREEN}Recorder shut down.{bcolors.ENDC}")
        
    if recorder_thread and recorder_thread.is_alive():
        recorder_thread.join()
        print(f"{bcolors.OKGREEN}Recorder thread finished.{bcolors.ENDC}")
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"{bcolors.OKGREEN}All async tasks cancelled.{bcolors.ENDC}")
    
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