import socket
import time
import io
import numpy as np
from PIL import Image

HOST = '0.0.0.0'  # Standard loopback interface address (localhost)
PORT = 5000  # Port to listen on (non-privileged ports are > 1023)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print('ready to accept incoming TCP connection')

        while True:
            conn, addr = s.accept()

            with conn:
                print('Connected by', addr)
                while True:
                    jpeg_file_len = recvall(conn, 4)
                    if not jpeg_file_len:
                        print('Cannot read jpeg length')
                        break
                    jpeg_file_len = int.from_bytes(jpeg_file_len, byteorder='big', signed=True)
                    print('JPEG file length is', jpeg_file_len)

                    depth_file_len = recvall(conn, 4)
                    if not depth_file_len:
                        print('Cannot read depth length')
                        break
                    depth_file_len = int.from_bytes(depth_file_len, byteorder='big', signed=True)
                    print('Depth file length is', depth_file_len)

                    start = time.time()
                    depth_file = recvall(conn, depth_file_len)
                    if not depth_file:
                        print('Cannot read depth')
                        break
                    np_pickle_file = io.BytesIO(depth_file)
                    depth = np.load(np_pickle_file)
                    print(depth.shape)

                    print('Reading depth file takes', time.time() - start)

                    start = time.time()
                    jpeg_file = recvall(conn, jpeg_file_len)
                    if not jpeg_file:
                        print('Cannot read jpeg')
                        break
                    jpeg_file = io.BytesIO(jpeg_file)
                    rgb = np.asarray(Image.open(jpeg_file))
                    print(rgb.shape)

                    print('Reading jpeg file takes', time.time() - start)

                    print('Read finished. Start sending.')

                    conn.sendall(depth_file)

                    print('Send complete')
