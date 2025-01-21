import socket

# Configuración del servidor UDP
UDP_IP = "192.168.1.102"  # Dirección IP del servidor
UDP_PORT = 8080           # Puerto UDP del servidor

# Crear el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Servidor UDP escuchando en {UDP_IP}:{UDP_PORT}")

while True:
    # Recibir datos del ESP32
    data, addr = sock.recvfrom(1024)  # Tamaño máximo del buffer: 1024 bytes
    print(f"Datos recibidos: {data} desde {addr}")
