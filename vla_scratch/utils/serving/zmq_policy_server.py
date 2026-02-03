import logging
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class ZmqPolicyServer:
    """ZMQ ROUTER server that handles multiple concurrent clients.

    Uses a ROUTER socket to accept connections from multiple clients simultaneously.
    Each request is queued and processed by the main loop, with responses routed
    back to the correct client using ZMQ's identity frames.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        max_queue_size: int = 100,
    ) -> None:
        self._host = host
        self._port = port or 0
        self._max_queue_size = max_queue_size

        self._context = zmq.Context.instance()
        self._socket: zmq.Socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 0)
        endpoint = self._endpoint()
        self._socket.bind(endpoint)
        logger.info("ZMQ ROUTER server bound at %s", endpoint)

        # Queue for incoming requests: (client_id, request_dict)
        self._request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        # Dict for outgoing responses: client_id -> response_dict
        self._response_dict: Dict[bytes, Dict[str, Any]] = {}
        self._response_lock = threading.Lock()

        self._stopped = False
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _endpoint(self) -> str:
        host = self._host
        if host in {"0.0.0.0", "::", ""}:
            host = "*"
        return f"tcp://{host}:{self._port}"

    def wait_for_request(
        self, timeout: float | None = None
    ) -> Optional[Dict[str, Any]]:
        """Block until a request is available, or timeout.

        Returns a dict with the request data, or None if stopped/timeout.
        """
        try:
            if timeout is None:
                request = self._request_queue.get()
            else:
                request = self._request_queue.get(timeout=timeout)
            return request
        except queue.Empty:
            return None

    def send_response(self, payload: Dict[str, Any]) -> None:
        """Send a response back to the client.

        The response is queued and will be sent by the background thread.
        The client_id is extracted from the payload (added by wait_for_request).
        """
        client_id = payload.pop("_client_id", None)
        if client_id is None:
            logger.warning("No client_id in response payload; dropping response.")
            return

        with self._response_lock:
            self._response_dict[client_id] = dict(payload)

    def close(self) -> None:
        self._stopped = True
        try:
            self._socket.close()
        finally:
            if hasattr(self, "_thread"):
                self._thread.join(timeout=1.0)

    def _serve_loop(self) -> None:
        """Background thread that handles ZMQ ROUTER socket I/O.

        Receives requests from multiple clients and queues them.
        Sends responses back to clients when available.
        """
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        try:
            while not self._stopped:
                # Poll with timeout to check for responses to send
                events = dict(poller.poll(timeout=100))  # 100ms timeout

                # Handle incoming requests
                if self._socket in events:
                    try:
                        frames = self._socket.recv_multipart(zmq.NOBLOCK)
                        # ROUTER socket: [client_id, empty_frame, ...payload_frames]
                        if len(frames) < 2:
                            logger.warning("Received malformed ROUTER frames")
                            continue

                        client_id = frames[0]
                        payload_frames = frames[2:]  # Skip empty delimiter

                        msg = _decode_request(payload_frames)
                        msg["_client_id"] = client_id  # Attach client ID for routing

                        # Queue the request for processing
                        try:
                            self._request_queue.put_nowait(msg)
                        except queue.Full:
                            logger.warning("Request queue full, dropping request")
                            # Send error response
                            error_resp = {"error": "Server queue full"}
                            self._send_to_client(client_id, error_resp)

                    except zmq.Again:
                        pass  # No message available
                    except Exception as e:
                        logger.error("Error receiving request: %s", e)

                # Send any pending responses
                with self._response_lock:
                    clients_to_remove = []
                    for client_id, response in list(self._response_dict.items()):
                        try:
                            self._send_to_client(client_id, response)
                            clients_to_remove.append(client_id)
                        except Exception as e:
                            logger.error("Error sending response to client: %s", e)
                            clients_to_remove.append(client_id)

                    for client_id in clients_to_remove:
                        self._response_dict.pop(client_id, None)

        finally:
            try:
                self._socket.close()
            except Exception:
                pass

    def _send_to_client(self, client_id: bytes, payload: Dict[str, Any]) -> None:
        """Send a response to a specific client via ROUTER socket."""
        frames = _encode_reply(payload)
        # ROUTER format: [client_id, empty_frame, ...payload_frames]
        self._socket.send_multipart([client_id, b""] + frames)


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


def _flatten_leaves(
    d: Dict[str, Any], prefix: List[str] | None = None
) -> List[Tuple[List[str], Any]]:
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


def _encode_reply(payload: Dict[str, Any]) -> List[bytes]:
    """Encode a response payload into multipart frames."""
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
    return [msgpack.packb(header)] + frames
