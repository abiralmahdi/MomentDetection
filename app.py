from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moment_detr.run_on_video.run import MomentDETRPredictor
import cv2

app = Flask(__name__, static_folder="static")

# Define directories
UPLOAD_FOLDER = "static/uploads"
CLIPPED_FOLDER = "static/clipped"
FRAME_FOLDER = "static/frames"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPPED_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Start and end time of the moment
start_time = 0
end_time = 0

# Load MomentDETR model
ckpt_path = "moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt"
predictor = MomentDETRPredictor(ckpt_path=ckpt_path, device="cpu")

# Global variables
video_path = None  # Store the uploaded video path
clipped_video = None  # Store the clipped video path
details = ""  # Store prediction details

def clip_video(video_path, start_time, end_time):
    """Clips the video between start_time and end_time."""
    clip = VideoFileClip(video_path).subclipped(start_time, end_time)
    output_path = os.path.join(CLIPPED_FOLDER, "clipped_video.mp4")
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    return output_path

def extract_frames(video_path):
    """Extract frames at 1-second intervals and save them with timestamps in the filename."""
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

    if frame_rate == 0:
        print("Error: Could not retrieve frame rate.")
        cap.release()
        return

    # Clear previous frames
    for file in os.listdir(FRAME_FOLDER):
        os.remove(os.path.join(FRAME_FOLDER, file))

    frame_count = 0
    saved_frame = 0
    interval = frame_rate  # Capture every 1 second

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Compute the current second based on frame count
        current_second = frame_count // frame_rate

        if frame_count % interval == 0:  # Save frame at every second
            frame_filename = os.path.join(FRAME_FOLDER, f'frame{current_second}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_frame += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame} frames successfully.")

@app.route("/")
def index():
    frames = [f"static/frames/{file}" for file in sorted(os.listdir(FRAME_FOLDER))]
    return render_template("index.html", video_path=video_path, clipped_video=clipped_video, prediction_details=details, frames=frames, start_frame="frame"+str(int(start_time)), end_frame="frame"+str(int(end_time)))

@app.route("/upload", methods=["POST"])
def upload_video():
    global video_path
    
    if "videoFile" not in request.files:
        return "No video file uploaded!", 400

    video_file = request.files["videoFile"]
    if video_file.filename == "":
        return "No selected file!", 400

    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # Extract frames after upload
    extract_frames(video_path)
    
    return redirect(url_for("index"))

@app.route("/query", methods=["POST"])
def process_query():
    global clipped_video, details
    
    query = request.form.get("queryInput")
    print(video_path)
    print(query)
    if not video_path:
        return "No video uploaded yet!", 400

    # Run MomentDETR prediction
    predictions = predictor.localize_moment(video_path=video_path, query_list=[query])
    start_time, end_time, score = predictions[0]['pred_relevant_windows'][0]

    # Clip video based on prediction
    clipped_video = clip_video(video_path, start_time, end_time)
    details = f"Moment found between {start_time:.2f} and {end_time:.2f} seconds (score: {score:.2f})."
    
    return redirect(url_for("index"))

@app.route("/download")
def download():
    return send_file("static/clipped/clipped_video.mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
