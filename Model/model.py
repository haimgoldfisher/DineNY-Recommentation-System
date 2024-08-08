from flask import Flask, jsonify
import subprocess
import threading
import schedule
import time

app = Flask(__name__)


def run_training():
    subprocess.call(["python3", "als_training.py"])


@app.route('/train/asl', methods=['POST'])
def train_model():
    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({"message": "ALS Training started"}), 202


def schedule_training():
    schedule.every(12).hours.do(run_training)
    while True:
        schedule.run_pending()
        time.sleep(1)


# Start the scheduling in a separate thread
scheduler_thread = threading.Thread(target=schedule_training)
scheduler_thread.daemon = True
scheduler_thread.start()

if __name__ == '__main__':
    app.run(debug=True)
