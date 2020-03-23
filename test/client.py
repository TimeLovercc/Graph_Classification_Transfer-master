from socket import *
import sys

addr = sys.argv[1]
port = sys.argv[2]
file = sys.argv[3]

ClientSocket = socket(AF_INET, SOCK_STREAM)
ClientSocket.connect((addr, int(port)))
head = 'GET /%s HTTP/1.1\r\n' % file
ClientSocket.send(head.encode())
ClientSocket.send(file.encode())
data = ClientSocket.recv(1024)
raw_data = ClientSocket.recv(1024)
data1 = ''
print(data.decode())
while raw_data :
    data1 = data1 + raw_data.decode()
    raw_data = ClientSocket.recv(1024)
data1 = data1 + raw_data.decode()
print(data1)
ClientSocket.close()

