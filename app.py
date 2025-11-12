from flask import Flask, jsonify, render_template, request
import threading, sounddevice as sd, numpy as np, librosa, pickle, time, os

app = Flask(__name__)

# global variables
whistle_count = 0
target_count = None
is_running = True
detection_enabled = False  # will start only after user sets target

# Load trained model
model = pickle.load(open("whistle_model.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    global whistle_count, target_count
    return jsonify({
        "count": whistle_count,
        "target": target_count if target_count is not None else 0
    })




@app.route("/set_target", methods=["POST"])
def set_target():
    global target_count, whistle_count, detection_enabled
    data = request.get_json()
    new_target = data.get("target", None)

    if not new_target:
        return jsonify({"error": "Target not provided"}), 400

    target_count = int(new_target)
    whistle_count = 0
    detection_enabled = True
    print(f" Target set to: {target_count}")
    return jsonify({"success": True, "target": target_count})


def detect_whistles():
    """Continuously listen for whistles once a target is set"""
    global whistle_count, is_running, detection_enabled, target_count

    fs = 44100
    last_detect_time = 0
    cooldown = 2.0  # seconds to prevent duplicate detections
    mic_index = sd.default.device[0]

    while is_running:
        # wait until target is set
        if not detection_enabled or not target_count:
            time.sleep(0.5)
            continue

        try:
            audio = sd.rec(int(2 * fs), samplerate=fs, channels=1, device=mic_index)
            sd.wait()
            y = audio.flatten()

            if y.size == 0 or not np.isfinite(y).all():
                continue

            y = y / (np.max(np.abs(y)) + 1e-9)
            mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)
            feat = np.mean(mfcc.T, axis=0).reshape(1, -1)
            pred = model.predict(feat)[0]

            now = time.time()
            if pred == 0 and now - last_detect_time > cooldown:
                whistle_count += 1
                last_detect_time = now
                print(f" Whistle detected! Total: {whistle_count}")

                if whistle_count >= target_count:
                    print("Target reached — stopping detection.")
                    detection_enabled = False

                    # Keep final values visible for frontend for a few seconds
                    reached_count = whistle_count
                    reached_target = target_count
                    print(f"Final count: {reached_count}/{reached_target}")

                    # Allow time for the browser to read final state
                    time.sleep(5)

                    # Reset values for next use
                    whistle_count = 0
                    target_count = None

        except Exception as e:
            print("Audio chunk skipped:", e)
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
            t = threading.Thread(target=detect_whistles, daemon=True)
            t.start()

        app.run(debug=True, port=5500, use_reloader=True)
    except KeyboardInterrupt:
        print("\n Stopping app gracefully...")
        is_running = False
        print(" Thread stopped.")
