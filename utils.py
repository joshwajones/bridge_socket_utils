import socket
from typing import Union


def receive_n_bytes(sock: socket.socket, n: int) -> Union[None, bytearray]:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
