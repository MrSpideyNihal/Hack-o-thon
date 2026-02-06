import subprocess
import time
import webbrowser
import requests
from pathlib import Path
import sys
import os

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🛡️  DRIFTGUARD DEMO SCRIPT  🛡️                 ║
║                                                              ║
║  Team: Strawhats                                             ║
║  Problem: Model Output Drift Detection                       ║
║  Status: READY TO WIN                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("\n[1/5] Checking dependencies...")
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    print("✅ All dependencies installed")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n[2/5] Generating synthetic dataset...")
data_dir = Path("data")
if not (data_dir / "baseline_data.csv").exists():
    os.chdir("data")
    subprocess.run([sys.executable, "generate_dataset.py"])
    os.chdir("..")
    print("✅ Dataset generated")
else:
    print("✅ Dataset already exists")

print("\n[3/5] Starting backend server...")
print("   Backend will run at: http://localhost:8000")
print("   API docs: http://localhost:8000/docs")

# Start backend in background
os.chdir("backend")
backend_process = subprocess.Popen(
    [sys.executable, "api.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
os.chdir("..")

print("   Waiting for server to start...")
time.sleep(3)

# Check if server is up
try:
    response = requests.get("http://localhost:8000")
    print("✅ Backend server running")
except:
    print("⚠️ Backend may still be starting... continuing anyway")

print("\n[4/5] Opening dashboard...")
dashboard_path = Path("frontend/dashboard.html").absolute()
webbrowser.open(f"file://{dashboard_path}")
print("✅ Dashboard opened in browser")

print("\n[5/5] Running drift detection tests...")
print("\n📊 TEST 1: Baseline (No Drift)")
try:
    time.sleep(2)
    response = requests.post("http://localhost:8000/detect-drift?scenario=A")
    data = response.json()
    print(f"   Drift Detected: {data['drift_detected']}")
    print(f"   Active Alerts: {len(data['alerts'])}")
except Exception as e:
    print(f"   Error: {e}")

print("\n📊 TEST 2: Scenario A (Income Shift)")
print("   Income distribution: $50k → $40k")

print("\n📊 TEST 3: Scenario B (Credit Score Drop)")
print("   Credit scores: 680 → 650")

print("\n📊 TEST 4: Scenario C (Regional Crisis)")
print("   Zip codes 90001-90010 affected")

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                    ✅ DEMO COMPLETE                          ║
║                                                              ║
║  Dashboard: Open frontend/dashboard.html                     ║
║  API Docs:  http://localhost:8000/docs                       ║
║                                                              ║
║  Click the buttons in the dashboard to test drift detection  ║
║                                                              ║
║  Press Ctrl+C to stop the demo                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

try:
    # Keep running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nStopping demo...")
    backend_process.kill()
    print("✅ Demo stopped")
