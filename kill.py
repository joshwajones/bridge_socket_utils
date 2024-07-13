import subprocess
import os
import argparse

DEFAULT_MODE = 'READ'
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


def kill():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', '-m', choices=['GREP', 'READ'], default=DEFAULT_MODE
    )
    args = parser.parse_args()
    mode = args.mode
    if mode == "GREP":
        for proc_name in proc_names:
            grep_and_kill(proc_name)
    elif mode == "READ":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(
                os.path.join(dir_path, 'pid_logs', 'overhead_pid.txt'), 'r'
        ) as file:
            overhead_pid = file.readline()
        subprocess.Popen(['kill', f'{overhead_pid}'])
        with open(
                os.path.join(dir_path, 'pid_logs', 'wrist_pid.txt'), 'r'
        ) as file:
            wrist_pid = file.readline()
        subprocess.Popen(['kill', f'{wrist_pid}'])
    else:
        raise ValueError(f'mode should be GREP or READ, received {mode}')




if __name__ == '__main__':
    kill()

