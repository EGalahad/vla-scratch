from typing import Any, Dict, Optional, List, Tuple

import zmq
import numpy as np
import msgpack


class ZmqPolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        rcv_timeout_ms: Optional[int] = None,
        snd_timeout_ms: Optional[int] = None,
    ) -> None:
        self._endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket = self._context.socket(zmq.REQ)
        if rcv_timeout_ms is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)
        if snd_timeout_ms is not None:
            self._socket.setsockopt(zmq.SNDTIMEO, snd_timeout_ms)
        # Avoid long blocking on close
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

    def close(self) -> None:
        if getattr(self, "_socket", None) is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None  # type: ignore

    def __enter__(self) -> "ZmqPolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401
        self.close()

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        req: Dict[str, Any] = {k: v for k, v in obs.items() if k != "type"}
        req["type"] = "infer"
        try:
            frames = _encode_raw_message(req)
            self._socket.send_multipart(frames)
            rframes = self._socket.recv_multipart()
            resp = _decode_raw_message(rframes)
        except zmq.Again as e:  # timeout
            raise TimeoutError("infer timed out waiting for reply") from e

        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        if not isinstance(resp, dict):
            raise RuntimeError("Invalid response from server")

        return resp

    def reset(self) -> None:
        try:
            frames = _encode_raw_message({"type": "reset"})
            self._socket.send_multipart(frames)
            rframes = self._socket.recv_multipart()
            resp = _decode_raw_message(rframes)
        except zmq.Again as e:
            raise TimeoutError("reset timed out waiting for reply") from e

        if isinstance(resp, dict) and resp.get("status") == "ok":
            return
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")

        raise RuntimeError("Invalid response to reset")


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


def _encode_raw_message(msg: Dict[str, Any]) -> List[bytes]:
    # Separate type and payload
    msg_type = msg.get("type")
    payload = {k: v for k, v in msg.items() if k != "type"}
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
            items.append({
                "path": path,
                "kind": "ndarray",
                "dtype": dtype_str,
                "shape": list(arr.shape),
            })
            frames.append(arr.tobytes())
        elif isinstance(value, str):
            items.append({
                "path": path,
                "kind": "str",
            })
            frames.append(value.encode("utf-8"))
        else:
            _assign_path(inline, path, value)

    header = {
        "format": "raw",
        "type": msg_type,
        "items": items,
        "inline": inline,
    }
    return [msgpack.packb(header)] + frames


def _decode_raw_message(frames: List[bytes]) -> Dict[str, Any]:
    if not frames:
        raise ValueError("empty multipart reply")
    header = msgpack.unpackb(frames[0])
    if not isinstance(header, dict) or header.get("format") != "raw":
        raise ValueError("invalid raw header")
    items = header.get("items", [])
    inline = header.get("inline", {})
    obj: Dict[str, Any] = dict(inline) if isinstance(inline, dict) else {}
    if len(frames) - 1 != len(items):
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
    return obj
