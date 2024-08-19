from fastapi import FastAPI, BackgroundTasks
import uvicorn
import subprocess
import threading
import schedule
import time
from datetime import datetime, timedelta

app = FastAPI()

# Global variable to track the next scheduled training time
next_training_time = datetime.now()

def run_training():
    global next_training_time
    # The training script should be executed here
    subprocess.call(["python3", "als_training.py"])
    # Update the next training time
    next_training_time = datetime.now() + timedelta(hours=3)  # Update based on the interval

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    # Start the training process in the background
    background_tasks.add_task(run_training)
    return {"message": "Training started"}

@app.get("/next-training")
async def get_next_training_time():
    global next_training_time
    now = datetime.now()
    time_remaining = next_training_time - now

    # Convert time_remaining to hours, minutes, and seconds
    total_seconds = int(time_remaining.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "time_left": f"Time left for next training: {hours} Hours, {minutes} Minutes, and {seconds} Seconds."
    }

def schedule_training():
    global next_training_time
    # Set the first training to occur 3 hours from now
    next_training_time = datetime.now() + timedelta(hours=3)

    # Schedule the training to run every 3 hours
    schedule.every(3).hours.do(run_training)

    # Wait until the first training time is reached
    while True:
        now = datetime.now()
        if now >= next_training_time:
            run_training()
            next_training_time = datetime.now() + timedelta(hours=3)  # Update to next schedule
        schedule.run_pending()
        time.sleep(1)

# Start the scheduling in a separate thread
scheduler_thread = threading.Thread(target=schedule_training)
scheduler_thread.daemon = True
scheduler_thread.start()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
