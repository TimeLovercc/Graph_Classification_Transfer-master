from socket import *
import sys

serverSocket = socket(AF_INET, SOCK_STREAM)

# Prepare a sever socket
host = 'localhost'
port = 12315
serverSocket.bind((host,port))
serverSocket.listen(10)

while True:
	# Establish the connection
	print (' The server is ready to receive')

	# Set up a new connection from the client
	connectionSocket, addr = serverSocket.accept()

	try:
		# Receives the request message from the client
		message =  connectionSocket.recv(1024)
		# Extract the path of the requested object from the message
		# The path is the second part of HTTP header, identified by [1]
		filename = message.split()[1]

		# Because the extracted path of the HTTP request includes 
		# a character '/', we read the path from the second character
		f = open(filename[1:])

		# Store the entire contenet of the requested file in a temporary buffer
		outputdata = f.read()

		# Send the HTTP response header line to the connection socket
		header = ' HTTP/1.1 200 OK\n\n'
		connectionSocket.send(header.encode())

		# Send the content of the requested file to the connection socket
		for i in range(0, len(outputdata)):
			connectionSocket.send(outputdata[i].encode())

		# Close the client connection socket
		connectionSocket.close()

	except IOError:
		# Send HTTP response message for file not found
		header = 'HTTP/1.1 404 Not Found\n\n'
		connectionSocket.send(header.encode())

		# Close the client connection socket
		connectionSocket.close()

serverSocket.close()

# Terminate the program after sending the corresponding data
sys.exit()
