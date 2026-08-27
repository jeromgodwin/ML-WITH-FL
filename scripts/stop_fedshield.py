"""FedShield — one-command stop. Kills backend (8000), frontend (3000), monitor."""
import subprocess, time

def kill_port(port):
    try:
        out = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True, text=True)
        for line in out.splitlines():
            parts = line.strip().split()
            if not parts: continue
            pid = parts[-1]
            if pid.isdigit():
                subprocess.call(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

if __name__ == "__main__":
    print("FedShield stopping...")
    kill_port(8000)
    kill_port(3000)
    # also kill monitor/frontend python/node by name as fallback
    subprocess.call('taskkill /F /IM python.exe 2>nul', shell=True)
    # don't kill all node — only vite (port 3000 already handled)
    time.sleep(1)
    print("Stopped. Check: netstat -an | findstr 8000")
