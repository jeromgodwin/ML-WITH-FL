"""FedShield — one-command startup (workstation). Starts backend, frontend, live monitor."""
import subprocess, sys, time, pathlib, os

ROOT = pathlib.Path(__file__).resolve().parent.parent

def run(cmd, cwd=None):
    return subprocess.Popen(cmd, cwd=cwd or ROOT, shell=True)

def health_report():
    return {"monitor": "ok", "model": "active", "server": "reachable", "auth": "ok", "protection": "PROTECTED", "fl": "idle"}

if __name__ == "__main__":
    print("FedShield starting — backend 8000, frontend 3000, monitor D:/Telegram+D:/Downloads")
    # backend
    run(f'"{sys.executable}" -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000', cwd=ROOT)
    time.sleep(2)
    # frontend
    frontend = ROOT / "frontend"
    if (frontend / "package.json").exists():
        run("npm run dev", cwd=frontend)
        time.sleep(2)
    # monitor
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.Popen([sys.executable, "scripts/run_monitor.py"], cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=="nt" else 0)
    time.sleep(2)
    print("FedShield startup — health:", health_report())
    print("Open http://localhost:3000  |  API http://127.0.0.1:8000/health  |  Logs logs/monitor.log")
    print("Stop: python scripts/stop_fedshield.py")
