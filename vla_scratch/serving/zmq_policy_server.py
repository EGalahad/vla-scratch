import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class ZmqPolicyServer:
    """Thin ZMQ REP server that only handles transport.

    A background thread receives raw multipart requests, exposes them via
    `wait_for_request`, and blocks until the main loop responds with
    `send_response`. This keeps policy inference in the caller (e.g. serve_policy)
    while maintaining the same over-the-wire protocol as before.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._host = host
        self._port = port or 0
        self._metadata = metadata or {}

        self._context = zmq.Context.instance()
        self._socket: zmq.Socket = self._context.socket(zmq.REP)
        self._socket.linger = 0
        endpoint = self._endpoint()
        self._socket.bind(endpoint)
        logger.info("ZMQ REP server bound at %s", endpoint)

        self._cv = threading.Condition()
        self._pending_request: Optional[Dict[str, Any]] = None
        self._pending_response: Optional[Dict[str, Any]] = None
        self._stopped = False

        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _endpoint(self) -> str:
        host = self._host
        if host in {"0.0.0.0", "::", ""}:
            host = "*"
        return f"tcp://{host}:{self._port}"

    def wait_for_request(self, timeout: float | None = None) -> Optional[Dict[str, Any]]:
        """Block until a request is available, or timeout."""
        with self._cv:
            if self._stopped:
                return None
            remaining = timeout
            while self._pending_request is None and not self._stopped:
                self._cv.wait(timeout=remaining)
                if timeout is not None:
                    remaining = 0.0
                    if self._pending_request is None:
                        break
            if self._stopped:
                return None
            return dict(self._pending_request) if self._pending_request is not None else None

    def send_response(self, payload: Dict[str, Any]) -> None:
        """Provide a response for the currently pending request.

        If no request is pending (e.g., due to a race or cancellation), we log and drop
        the payload instead of raising to avoid crashing the caller loop.
        """
        with self._cv:
            if self._pending_request is None:
                logger.warning("No pending request to respond to; dropping response.")
                return
            self._pending_response = dict(payload)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        try:
            self._socket.close()
        finally:
            if hasattr(self, "_thread"):
                self._thread.join(timeout=1.0)

    def _serve_loop(self) -> None:
        try:
            while not self._stopped:
                try:
                    frames = self._socket.recv_multipart()
                except zmq.ZMQError:
                    break
                msg = _decode_request(frames)

                with self._cv:
                    while self._pending_request is not None and not self._stopped:
                        self._cv.wait()
                    if self._stopped:
                        break
                    self._pending_request = msg
                    self._cv.notify_all()
                    while self._pending_response is None and not self._stopped:
                        self._cv.wait()
                    if self._stopped:
                        break
                    response = self._pending_response
                    self._pending_response = None

                if response is not None:
                    _send_reply(self._socket, response)

                with self._cv:
                    self._pending_request = None
                    self._cv.notify_all()
        finally:
            try:
                self._socket.close()
            except Exception:
                pass


def _decode_request(frames: List[bytes]) -> Dict[str, Any]:
    """Decode raw multipart request frames into a dict."""
    if not frames:
        raise ValueError("empty request frames")
    header = msgpack.unpackb(frames[0])
    if not isinstance(header, dict) or header.get("format") != "raw":
        raise ValueError("invalid raw header")
    items = header.get("items", [])
    inline = header.get("inline", {})
    msg_type = header.get("type")
    obj: Dict[str, Any] = dict(inline) if isinstance(inline, dict) else {}
    expected = len(items)
    if len(frames) - 1 != expected:
        raise ValueError("raw frames count mismatch")
    for idx, spec in enumerate(items):
        path = spec.get("path")
        kind = spec.get("kind")
        if not isinstance(path, list):
            raise TypeError("path must be a list of keys")
        data = frames[idx + 1]
        if kind == "ndarray":
            shape = tuple(spec.get("shape", ()))
            dtype_str = spec.get("dtype", "float32")
            if dtype_str == "float32":
                dtype = np.float32
            elif dtype_str == "uint8":
                dtype = np.uint8
            else:
                dtype = np.dtype(dtype_str)
            arr = np.frombuffer(data, dtype=dtype)
            if shape:
                arr = arr.reshape(shape)
            _assign_path(obj, path, arr)
        elif kind == "str":
            s = data.decode("utf-8")
            _assign_path(obj, path, s)
        else:
            raise ValueError(f"unknown kind: {kind}")

    if msg_type is not None:
        obj["type"] = msg_type
    return obj


def _flatten_leaves(d: Dict[str, Any], prefix: List[str] | None = None) -> List[Tuple[List[str], Any]]:
    if prefix is None:
        prefix = []
    leaves: List[Tuple[List[str], Any]] = []
    for k, v in d.items():
        key = str(k)
        if isinstance(v, dict):
            leaves.extend(_flatten_leaves(v, prefix + [key]))
        else:
            leaves.append((prefix + [key], v))
    return leaves


def _assign_path(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _send_reply(socket: zmq.Socket, payload: Dict[str, Any]) -> None:
    items: List[Dict[str, Any]] = []
    frames: List[bytes] = []
    inline: Dict[str, Any] = {}
    for path, value in _flatten_leaves(payload):
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8:
                arr = np.ascontiguousarray(value, dtype=np.uint8)
                dtype_str = "uint8"
            else:
                arr = np.ascontiguousarray(value, dtype=np.float32)
                dtype_str = "float32"
            items.append(
                {
                    "path": path,
                    "kind": "ndarray",
                    "dtype": dtype_str,
                    "shape": list(arr.shape),
                }
            )
            frames.append(arr.tobytes())
        elif isinstance(value, str):
            items.append(
                {
                    "path": path,
                    "kind": "str",
                }
            )
            frames.append(value.encode("utf-8"))
        else:
            _assign_path(inline, path, value)

    header = {
        "format": "raw",
        "type": payload.get("type", "infer_result"),
        "items": items,
        "inline": inline,
    }
    socket.send_multipart([msgpack.packb(header)] + frames)
