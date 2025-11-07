# run_live_app.py
# This file will run your LIVE webcam app and make it public

import os
import subprocess
from pyngrok import ngrok, conf
import time
import sys

# --- IMPORTANT: NGrok Authtoken ---
# You already did this, so you are all set.
# ----------------------------------

print("Starting ngrok tunnel for Streamlit app (port 8501)...")

# Connect to ngrok and create a tunnel to port 8501
# Streamlit's default port is 8501
try:
    public_url = ngrok.connect(8501)
except Exception as e:
    print(f"Error starting ngrok: {e}")
    print("Please make sure you have added your authtoken.")
    sys.exit(1)

print("=" * 70)
print(f"✅  Your LIVE Interview App is ready!")
print(f"➡️  Send THIS link to your manager: {public_url}")
print("=" * 70)

# Start the Streamlit app
print("Starting your Streamlit app (main_app.py)...")
print("Your app will open in a new browser window. You can ignore it.")
print("The PUBLIC link is the one above.")

# We use subprocess.Popen to run Streamlit
# 'main_app.py' is your live webcam project
process = subprocess.Popen(["streamlit", "run", "main_app.py", "--server.port", "8501"])

try:
    # Keep the script alive while the Streamlit process is running
    process.wait()
except KeyboardInterrupt:
    print("\nShutting down Streamlit and ngrok...")
    ngrok.disconnect(public_url)
    process.terminate()