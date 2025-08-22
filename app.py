from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import fitz
from gtts import gTTS
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

from moviepy.video.VideoClip import ColorClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
import os
import shutil
import uuid
import random

app = FastAPI()
app.mount("/resources", StaticFiles(directory="resources"), name="resources")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# List of Gen-Z style intros
GENZ_INTROS = [
    "Ayo rizzler, we gonna learn about...",
    "No cap, this is about to be fire...",
    "Bet you didn't know this...",
    "Slay alert! Let's get into...",
    "This is giving main character energy...",
    "Lowkey obsessed with this...",
    "This is absolutely bussin...",
    "Sup fam, this is about to blow your mind...",
    "It's giving educational vibes...",
    "Let's get this bread and learn about..."
]

def extract_text(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    if ext == "pdf":
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif ext == "txt":
        return file.file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        summary = summarizer(text, max_length=500, min_length=100, do_sample=False)[0]['summary_text']
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips

from faster_whisper import WhisperModel
import tempfile

@app.post("/generate-video")
async def generate_video(file: UploadFile = File(...)):
    try:
        # SET THE CORRECT IMAGEMAGICK PATH
        os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

        # Extract text and summarize
        text = extract_text(file)
        summary = summarizer(text, max_length=500, min_length=100, do_sample=False)[0]['summary_text']

        # Add Gen-Z intro
        genz_intro = random.choice(GENZ_INTROS)
        full_text = f"{genz_intro} {summary}"

        # Create TTS audio
        audio_path = f"static/{uuid.uuid4()}.mp3"
        gTTS(text=full_text, lang='en').save(audio_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        # Load background video and loop to match audio duration
        bg_path = "static/videos/subwaysufervideoloop.mp4"
        bg_clip = VideoFileClip(bg_path)
        
        # Calculate how many loops we need
        loops_needed = int(duration / bg_clip.duration) + 1
        bg_clip = bg_clip.loop(n=loops_needed)
         
        # Trim to exact audio duration and resize
        bg_clip = bg_clip.subclip(0, duration).resize((720, 480))

        # ====== IMPROVED KARAOKE STYLE CAPTIONS ======
        # Use Whisper to get word-level timings
        model = WhisperModel("base")
        
        # Transcribe the audio to get word timings
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        
        # Collect all words with their start and end times
        word_timings = []
        for segment in segments:
            for word in segment.words:
                word_timings.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
        
        caption_clips = []
        
        # Style the intro differently (first few words)
        intro_end_index = len(genz_intro.split())
        
        for i, word_info in enumerate(word_timings):
            word = word_info["word"]
            start_time = word_info["start"]
            end_time = word_info["end"]
            word_duration = end_time - start_time
            
            # Different styling for intro vs content
            if i < intro_end_index:
                # Intro words - more flashy
                font_size = 54
                color = "#FF6B6B"  # Bright red
                stroke_color = "#4ECDC4"  # Teal outline
                font = "Impact"
            else:
                # Content words - regular style
                font_size = 48
                color = "white"
                stroke_color = "black"
                font = "Arial-Bold"
            
            # Create text clip for each word
            txt_clip = TextClip(
                word,
                fontsize=font_size,
                color=color,
                font=font,
                stroke_color=stroke_color,
                stroke_width=2,
                method="label",
            ).set_position("center").set_start(start_time).set_duration(word_duration)
            
            caption_clips.append(txt_clip)

        # Overlay captions on background video
        final_video = CompositeVideoClip([bg_clip, *caption_clips])
        final_video = final_video.set_audio(audio_clip)

        # Save video with proper codec settings
        video_path = f"static/{uuid.uuid4()}.mp4"
        final_video.write_videofile(
            video_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            threads=4,
            preset="medium",
            ffmpeg_params=["-crf", "23"]
        )

        # Cleanup
        audio_clip.close()
        bg_clip.close()
        final_video.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {"videoUrl": f"/static/{os.path.basename(video_path)}", "summary": summary}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
