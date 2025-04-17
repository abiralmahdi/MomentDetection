from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moment_detr.run_on_video.run import MomentDETRPredictor

ckpt_path = "moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt"
predictor = MomentDETRPredictor(ckpt_path=ckpt_path, device="cpu")

video_path = "static/uploads/RoripwjYFp8_60.0_210.0.mp4"
query = "anomaly detection"

predictions = predictor.localize_moment(video_path=video_path, query_list=[query])
print(predictions)