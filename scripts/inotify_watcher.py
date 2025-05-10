#!/usr/bin/env python3
import os
import sys
import struct
import select

IN_MODIFY = 0x00000002

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file_to_watch> <timeout_seconds>", file=sys.stderr)
        sys.exit(2)
    filepath = sys.argv[1]
    timeout = float(sys.argv[2])

    fd = os.inotify_init()
    if fd < 0:
        print("Failed to initialize inotify", file=sys.stderr)
        sys.exit(2)

    wd = os.inotify_add_watch(fd, filepath, IN_MODIFY)
    if wd < 0:
        print(f"Failed to add watch on {filepath}", file=sys.stderr)
        os.close(fd)
        sys.exit(2)

    poller = select.poll()
    poller.register(fd, select.POLLIN)

    events = []
    while True:
        events = poller.poll(timeout * 1000)
        if not events:
            # Timeout occurred
            os.close(fd)
            sys.exit(1)
        else:
            # Read events
            data = os.read(fd, 4096)
            i = 0
            while i < len(data):
                wd, mask, cookie, length = struct.unpack_from('iIII', data, i)
                i += struct.calcsize('iIII')
                if length > 0:
                    name = data[i:i+length].rstrip(b'\0')
                    i += length
                else:
                    name = b''
                if mask & IN_MODIFY:
                    os.close(fd)
                    sys.exit(0)
            # Continue waiting if no modify event found

if __name__ == '__main__':
    main()
    
