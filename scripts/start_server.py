"""Start the local server independently of the launching terminal on Windows."""
import os
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
data = root / 'data'
data.mkdir(exist_ok=True)
command = [sys.executable, '-m', 'uvicorn', 'backend.main:app',
           '--host', '127.0.0.1', '--port', '8765']
options = {'cwd': root, 'stdin': subprocess.DEVNULL}
if os.name == 'nt':
    # A terminal/automation job may terminate its entire process tree on exit.
    # Detach the server and explicitly leave that job when Windows permits it.
    options['creationflags'] = (subprocess.DETACHED_PROCESS |
                               subprocess.CREATE_NEW_PROCESS_GROUP |
                               subprocess.CREATE_BREAKAWAY_FROM_JOB)
else:
    options['start_new_session'] = True

with (data / 'server.stdout.log').open('ab', buffering=0) as stdout, \
        (data / 'server.stderr.log').open('ab', buffering=0) as stderr:
    try:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **options)
    except OSError as exc:
        if os.name != 'nt' or exc.winerror != 5:
            raise
        # Some hosts prohibit job breakaway. The desktop launcher can still run
        # normally; make the limitation visible instead of claiming persistence.
        print('This host restricts independent processes. If the server stops, '
              'launch Start-Studio.cmd directly from File Explorer.', file=sys.stderr)
        options['creationflags'] &= ~subprocess.CREATE_BREAKAWAY_FROM_JOB
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **options)
print(f'Started Studio launcher PID {process.pid}.')
