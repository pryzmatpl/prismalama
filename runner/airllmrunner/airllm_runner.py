#!/usr/bin/env python3
"""
AirLLM Runner - Ollama-compatible HTTP server for AirLLM models.
Provides the same API as llama runner but uses AirLLM for layer-by-layer inference.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    LAUNCHED = "launched"
    LOADING_MODEL = "loadingModel"
    READY = "ready"


class DoneReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    CONNECTION_CLOSED = "connectionClosed"


@dataclass
class LoadRequest:
    operation: str
    model_path: str = ""
    lora_path: List[str] = field(default_factory=list)
    projector_path: str = ""
    parallel: int = 1
    batch_size: int = 512
    kv_size: int = 4096
    kv_cache_type: str = ""
    flash_attention: str = "auto"
    num_threads: int = 4
    multi_user_cache: bool = False
    gpu_layers: List[Dict] = field(default_factory=list)
    main_gpu: int = 0
    use_mmap: bool = True


@dataclass
class LoadResponse:
    success: bool = False
    error: str = ""


@dataclass
class ServerStatusResponse:
    status: str
    progress: float = 0.0


@dataclass
class CompletionRequest:
    prompt: str
    images: List[Dict] = field(default_factory=list)
    grammar: str = ""
    options: Dict = field(default_factory=dict)
    logprobs: bool = False
    top_logprobs: int = 0
    shift: bool = True
    truncate: bool = True


@dataclass
class CompletionResponse:
    content: str = ""
    logprobs: List = field(default_factory=list)
    done: bool = False
    done_reason: str = ""
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


@dataclass
class EmbeddingRequest:
    content: str


@dataclass
class EmbeddingResponse:
    embedding: List[float] = field(default_factory=list)
    prompt_eval_count: int = 0


class AirLLMModel:
    """Wrapper for AirLLM models with ROCm support."""
    
    def __init__(self):
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_path: Optional[str] = None
        self.status = ServerStatus.LAUNCHED
        self.progress = 0.0
        self.lock = threading.Lock()
        self.compression = "4bit"
        self._model_loaded = False
        
    def load(self, model_path: str, options: dict):
        """Load model using AirLLM with layer-by-layer loading."""
        self.status = ServerStatus.LOADING_MODEL
        self.progress = 0.0
        self.model_path = model_path
        
        def load_thread():
            try:
                sys.path.insert(0, "/usr/share/ollama/airllm")
                
                logger.info(f"Loading AirLLM model from {model_path}")
                self.progress = 0.1
                
                from airllm import AutoModel
                import torch
                
                # Extract options
                compression = options.get('compression', os.environ.get('AIRLLM_COMPRESSION', '4bit'))
                main_gpu = options.get('main_gpu', 0)
                kv_size = options.get('kv_size', 2048)
                
                # Set device
                if torch.cuda.is_available():
                    device = f"cuda:{main_gpu}"
                else:
                    device = "cpu"
                
                cuda_ok = torch.cuda.is_available()
                n_dev = torch.cuda.device_count() if cuda_ok else 0
                logger.info(
                    "PyTorch: cuda.is_available=%s device_count=%s device=%s",
                    cuda_ok,
                    n_dev,
                    device,
                )
                if not cuda_ok:
                    logger.warning(
                        "PyTorch reports no CUDA/ROCm GPU; AirLLM will run on CPU unless you install "
                        "a ROCm-enabled PyTorch (e.g. python-pytorch-rocm on Arch) and set HIP / "
                        "AIRLLM_DEVICE correctly."
                    )
                
                self.progress = 0.3
                logger.info("AutoModel imported, initializing...")
                
                # AutoModel uses from_pretrained classmethod
                self.model = AutoModel.from_pretrained(
                    model_path,
                    device=device,
                    compression=compression,
                    max_seq_len=kv_size,
                    profiling_mode=False
                )
                self.progress = 0.5
                logger.info("AutoModel initialized")
                
                # Tokenizer and model are already loaded by from_pretrained
                self.tokenizer = self.model.tokenizer
                self.progress = 0.7
                
                # Model is already initialized by from_pretrained
                self.progress = 1.0
                self._model_loaded = True
                
                self.status = ServerStatus.READY
                logger.info("AirLLM model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.error(traceback.format_exc())
                self.status = ServerStatus.LAUNCHED
                self.progress = 0.0
                self._model_loaded = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
        
    def is_ready(self) -> bool:
        return self.status == ServerStatus.READY
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using AirLLM."""
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        max_new_tokens = kwargs.get('num_predict', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        stop = kwargs.get('stop', [])
        
        if isinstance(stop, str):
            stop = [stop]
        
        try:
            result = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop
            )
            return result
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
        finally:
            if os.environ.get("AIRLLM_POST_INFER_CLEANUP", "1") != "0":
                try:
                    from airllm.utils import finalize_inference_memory
                    finalize_inference_memory()
                except Exception as ex:
                    logger.debug("finalize_inference_memory: %s", ex)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())


class AirLLMHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing ollama runner API."""
    
    model: Optional[AirLLMModel] = None
    protocol_version = 'HTTP/1.1'
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")
    
    def send_json_response(self, data: dict, status: int = 200):
        response = json.dumps(data)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        return response
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, 400)
            return
        
        if self.path == '/load':
            self.handle_load(data)
        elif self.path == '/completion':
            self.handle_completion(data)
        elif self.path == '/embedding':
            self.handle_embedding(data)
        else:
            self.send_json_response({'error': 'Not found'}, 404)
    
    def do_GET(self):
        if self.path == '/health':
            self.handle_health()
        else:
            self.send_json_response({'error': 'Not found'}, 404)
    
    def handle_load(self, data: dict):
        """Handle model load request."""
        try:
            if self.model is None:
                self.send_json_response({'success': False, 'error': 'Model not initialized'})
                return
                
            req = LoadRequest(**data)
            
            if req.operation == "commit":
                model_path = req.model_path
                if not model_path:
                    self.send_json_response({'success': False, 'error': 'No model path'})
                    return
                
                options = {
                    'compression': os.environ.get('AIRLLM_COMPRESSION', '4bit'),
                    'main_gpu': req.main_gpu,
                    'kv_size': req.kv_size,
                }
                self.model.load(model_path, options)
                self.send_json_response({'success': True})
                
            elif req.operation == "close":
                self.send_json_response({'success': True})
            else:
                self.send_json_response({'success': True})
                
        except Exception as e:
            logger.error(f"Load error: {e}")
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_completion(self, data: dict):
        """Handle completion request with streaming."""
        try:
            if self.model is None:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Model not initialized'}).encode())
                return
                
            req = CompletionRequest(**data)
            
            if not self.model.is_ready():
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Model not ready'}).encode())
                return
            
            options = req.options or {}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            
            prompt_tokens = self.model.count_tokens(req.prompt)
            start_time = time.time()
            
            try:
                result = self.model.generate(
                    req.prompt,
                    num_predict=options.get('num_predict', 512),
                    temperature=options.get('temperature', 0.7),
                    top_p=options.get('top_p', 0.9),
                    stop=options.get('stop', [])
                )
                
                gen_duration = int((time.time() - start_time) * 1e9)
                gen_tokens = self.model.count_tokens(result)
                
                response = CompletionResponse(
                    content=result,
                    done=True,
                    done_reason=DoneReason.STOP.value,
                    prompt_eval_count=prompt_tokens,
                    prompt_eval_duration=int(0.1 * 1e9),
                    eval_count=gen_tokens,
                    eval_duration=gen_duration
                )
                
                chunk = json.dumps(asdict(response))
                self.wfile.write(f"{len(chunk):x}\r\n{chunk}\r\n".encode())
                self.wfile.write("0\r\n\r\n".encode())
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                error_resp = CompletionResponse(
                    done=True,
                    done_reason=DoneReason.STOP.value
                )
                chunk = json.dumps(asdict(error_resp))
                self.wfile.write(f"{len(chunk):x}\r\n{chunk}\r\n".encode())
                self.wfile.write("0\r\n\r\n".encode())
                
        except Exception as e:
            logger.error(f"Completion error: {e}")
            logger.error(traceback.format_exc())
    
    def handle_embedding(self, data: dict):
        """Handle embedding request."""
        try:
            if self.model is None:
                self.send_json_response({'error': 'Model not initialized'}, 500)
                return
                
            req = EmbeddingRequest(**data)
            
            response = EmbeddingResponse(
                embedding=[0.0] * 768,
                prompt_eval_count=self.model.count_tokens(req.content)
            )
            
            self.send_json_response(asdict(response))
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            self.send_json_response({'error': str(e)}, 500)
    
    def handle_health(self):
        """Handle health check."""
        response = ServerStatusResponse(
            status=self.model.status.value if self.model else ServerStatus.LAUNCHED.value,
            progress=self.model.progress if self.model else 0.0
        )
        self.send_json_response(asdict(response))


def _require_airllm_python_deps():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        logger.critical(
            "AirLLM requires Python package 'transformers'. On Arch: "
            "sudo python3 -m pip install --break-system-packages transformers safetensors "
            "OR yay -S python-transformers python-safetensors (AUR); keep python-pytorch-rocm from repos."
        )
        raise SystemExit(2) from e


def main():
    parser = argparse.ArgumentParser(description='AirLLM Runner for Ollama')
    parser.add_argument('--model', type=str, default='', help='Path to model')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _require_airllm_python_deps()
    
    AirLLMHandler.model = AirLLMModel()
    
    server = HTTPServer(('127.0.0.1', args.port), AirLLMHandler)
    logger.info(f"AirLLM Runner listening on 127.0.0.1:{args.port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
