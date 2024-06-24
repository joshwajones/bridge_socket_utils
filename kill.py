import subprocess


def grep_and_kill(proc_cmd):
    proclist_proc = subprocess.Popen(['ps', '-e'], stdout=subprocess.PIPE)
    grep_proc = subprocess.Popen(['grep', proc_cmd], stdin=proclist_proc.stdout,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    proclist_proc.stdout.close()
    out, _ = grep_proc.communicate()
    out = str(out)[2:-1]
    lines = out.split('\\n')

    for line in lines:
        line = line.strip()
        if line:
            pid = line[:line.index(' ')]
            subprocess.Popen(['kill', f'{pid}'])


proc_names = ['python imu_stream.py', 'python video_stream.py']
# proc_names = ['python video_stream.py']
if __name__ == '__main__':
    for proc_name in proc_names:
        grep_and_kill(proc_name)
