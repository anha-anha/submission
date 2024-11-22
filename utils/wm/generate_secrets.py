from Crypto.Random import get_random_bytes

def write_file(filename, array_name, length, count=10000):
    with open(filename, 'w') as file:
        file.write(f"{array_name} = [\n")
        for _ in range(count):
            random_bytes = get_random_bytes(length)
            # Get the repr of bytes, which Python can later evaluate to bytes
            byte_string = repr(random_bytes)
            file.write(f'    {byte_string},\n')
        file.write("]\n")

# Create the files with specific byte lengths
write_file('messages.py', 'MESSAGES', 1024)
write_file('keys.py', 'KEYS', 32)
write_file('nonces.py', 'NONCES', 16)