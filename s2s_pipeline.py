import logging
import os
import socket
import sys
import threading
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import perf_counter

import openai
import requests

import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    TextIteratorStreamer
)


from utils import (
    VADIterator,
    int2float,
    next_power_of_2
)

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp") 
torch._inductor.config.fx_graph_cache = True
# mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
torch._dynamo.config.cache_size_limit = 15


console = Console()


@dataclass
class ModuleArguments:
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=warning."
        }
    )


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(self):
        pass

    def process(self):
        raise NotImplementedError

    def run(self):
        while not self.stop_event.is_set():
            input = self.queue_in.get()
            if isinstance(input, bytes) and input == b'END':
                # sentinelle signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            for output in self.process(input):
                self._times.append(perf_counter() - start_time)
                logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                self.queue_out.put(output)
                start_time = perf_counter()

        self.cleanup()
        self.queue_out.put(b'END')

    @property
    def last_time(self):
        return self._times[-1]

    def cleanup(self):
        pass


@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP ddress for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    recv_port: int = field(
        default=12345,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )
    chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
        }
    )


class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self, 
        stop_event,
        queue_out,
        should_listen,
        host='0.0.0.0', 
        port=12345,
        chunk_size=1024
    ):  
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size=chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b''
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                # connection closed
                return None  
            data += packet
        return data

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Receiver waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b'END')
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")


@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    send_port: int = field(
        default=12346,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )

            
class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(
        self, 
        stop_event,
        queue_in,
        host='0.0.0.0', 
        port=12346
    ):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Sender waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b'END':
                break
        self.conn.close()
        logger.info("Sender closed")


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.3,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        }
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        }
    )
    min_silence_ms: int = field(
        default=250,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 1000 ms."
        }
    )
    min_speech_ms: int = field(
        default=750,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        }
    )
    max_speech_ms: float = field(
        default=float('inf'),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        }
    )
    speech_pad_ms: int = field(
        default=30,
        metadata={
            "help": "Amount of padding added to the beginning and end of detected speech segments. Measured in milliseconds. Default is 30 ms."
        }
    )


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
            self, 
            should_listen,
            thresh=0.3, 
            sample_rate=16000, 
            min_silence_ms=1000,
            min_speech_ms=500, 
            max_speech_ms=float('inf'),
            speech_pad_ms=30,

        ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
            else:
                self.should_listen.clear()
                logger.debug("Stop listening")
                yield array


@dataclass
class WhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        }
    )
    stt_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        } 
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        }
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        }
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        }
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        }
    )
    stt_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        }
    )
    stt_gen_language: str = field(
        default="en",
        metadata={
            "help": "The language of the speech to transcribe. Default is 'en' for English."
        }
    )


class WhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
            self,
            model_name="distil-whisper/distil-large-v3",
            device="cuda",  
            torch_dtype="float16",  
            compile_mode=None,
            gen_kwargs={}
        ): 
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode=compile_mode
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)
        
        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        self.warmup()
    
    def prepare_model_inputs(self, spoken_prompt):
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features
        
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture 
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1,  self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device
        ) 
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        input_features = self.prepare_model_inputs(spoken_prompt)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=False
        )[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text


@dataclass
class OpenAILanguageModelHandlerArguments:
    openai_api_key: str = field(
        default="",
        metadata={"help": "OpenAI API key"}
    )
    model: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "OpenAI model to use"}
    )
    max_tokens: int = field(
        default=150,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature"}
    )
    chat_size: int = field(
        default=1,
        metadata={"help": "Number of interactions assistant-user to keep for the chat. None for no limitations."}
    )

class OpenAILanguageModelHandler(BaseHandler):
    def setup(
        self,
        openai_api_key,
        model="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.7,
        chat_size=1,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        openai.api_key = openai_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat = Chat(chat_size)
        if init_chat_prompt:
            self.chat.init_chat(
                {"role": "system", "content": init_chat_prompt}
            )

    def process(self, prompt):
        logger.debug("Inferring using OpenAI API...")

        self.chat.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.chat.to_list(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True
        )

        generated_text = ""
        for chunk in response:
            if chunk.choices[0].delta.get("content"):
                new_text = chunk.choices[0].delta.content
                generated_text += new_text
                yield new_text

        self.chat.append({"role": "assistant", "content": generated_text})
        logger.debug("Finished OpenAI API inference")


@dataclass
class ElevenlabsTTSHandlerArguments:
    elevenlabs_api_key: str = field(
        default="",
        metadata={"help": "Elevenlabs API key"}
    )
    voice_id: str = field(
        default="21m00Tcm4TlvDq8ikWAM",
        metadata={"help": "Elevenlabs voice ID"}
    )
    model_id: str = field(
        default="eleven_monolingual_v1",
        metadata={"help": "Elevenlabs model ID"}
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "Size of audio chunks to stream"}
    )

class ElevenlabsTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        elevenlabs_api_key,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_monolingual_v1",
        chunk_size=1024
    ):
        self.should_listen = should_listen
        self.api_key = elevenlabs_api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.chunk_size = chunk_size

    def process(self, text):
        console.print(f"[green]ASSISTANT: {text}")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers, stream=True)
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    yield chunk
        else:
            logger.error(f"Error in TTS API call: {response.status_code}")
            yield b""

        self.should_listen.set()


def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """

    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def main():
    parser = HfArgumentParser((
        ModuleArguments,
        SocketReceiverArguments, 
        SocketSenderArguments,
        VADHandlerArguments,
        WhisperSTTHandlerArguments,
        OpenAILanguageModelHandlerArguments,
        ElevenlabsTTSHandlerArguments,
    ))

    # 0. Parse CLI arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        (
            module_kwargs,
            socket_receiver_kwargs, 
            socket_sender_kwargs, 
            vad_handler_kwargs, 
            whisper_stt_handler_kwargs, 
            language_model_handler_kwargs, 
            parler_tts_handler_kwargs,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        (
            module_kwargs,
            socket_receiver_kwargs, 
            socket_sender_kwargs, 
            vad_handler_kwargs, 
            whisper_stt_handler_kwargs, 
            language_model_handler_kwargs, 
            parler_tts_handler_kwargs,
        ) = parser.parse_args_into_dataclasses()

    # 1. Handle logger
    global logger
    logging.basicConfig(
        level=module_kwargs.log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if module_kwargs.log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    # 2. Prepare each part's arguments
    prepare_args(whisper_stt_handler_kwargs, "stt")
    prepare_args(openai_lm_handler_kwargs, "openai")
    prepare_args(elevenlabs_tts_handler_kwargs, "elevenlabs")

    # 3. Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event() 
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue() 
    text_prompt_queue = Queue()
    lm_response_queue = Queue()
    
    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
    stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs=vars(whisper_stt_handler_kwargs),
    )
    lm = OpenAILanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=vars(openai_lm_handler_kwargs),
    )
    tts = ElevenlabsTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(elevenlabs_tts_handler_kwargs),
    )

    recv_handler = SocketReceiver(
        stop_event, 
        recv_audio_chunks_queue, 
        should_listen,
        host=socket_receiver_kwargs.recv_host,
        port=socket_receiver_kwargs.recv_port,
        chunk_size=socket_receiver_kwargs.chunk_size,
    )

    send_handler = SocketSender(
        stop_event, 
        send_audio_chunks_queue,
        host=socket_sender_kwargs.send_host,
        port=socket_sender_kwargs.send_port,
        )

    # 4. Run the pipeline
    try:
        pipeline_manager = ThreadManager([vad, tts, lm, stt, recv_handler, send_handler])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()
    
if __name__ == "__main__":
    main()
