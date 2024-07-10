import subprocess
import os

def grep_and_kill(proc_cmd):
    proclist_proc = subprocess.Popen(['ps', '-e'], stdout=subprocess.PIPE)
    grep_proc = subprocess.Popen(['grep', proc_cmd], stdin=proclist_proc.stdout,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    proclist_proc.stdout.close()
    out, _ = grep_proc.communicate()
    out = str(out)[2:-1]
    lines = out.split('\\n')
    curr_pid = os.getpid()
    pids = []
    for line in lines:
        line = line.strip()
        if line:
            pid = line[:line.index(' ')]
            if pid != curr_pid:
                subprocess.Popen(['kill', f'{pid}'])


proc_names = [
    'python bag_overhead_stream.py',
    'python wrist_stream.py',
    'python flat_overhead_stream.py',
]
if __name__ == '__main__':
    for proc_name in proc_names:
        grep_and_kill(proc_name)
