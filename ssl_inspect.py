import socket
import ssl

hostname = "4660f290.databases.neo4j.io"

context = ssl._create_unverified_context()

with socket.create_connection((hostname, 7687)) as sock:
    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
        cert = ssock.getpeercert(binary_form=False)
        print(cert)