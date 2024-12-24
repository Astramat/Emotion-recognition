from flask import Flask, request, render_template_string, Response
import subprocess
from googleapiclient.discovery import build
import random

app = Flask(__name__)

API_KEY = 'Pa'  # Remplace par ta clé API YouTube
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stream YouTube Audio</title>
</head>
<body>
    <form method="post">
        <label for="keyword">Keyword:</label>
        <input type="text" id="keyword" name="keyword" required>
        <input type="submit" value="Stream Audio">
    </form>
    {% if audio_url %}
        <audio controls autoplay style="display:none;">
            <source src="{{ audio_url }}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    {% endif %}
</body>
</html>
'''

def search_youtube(keyword):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    search_response = youtube.search().list(
        q=keyword,
        part='id,snippet',
        maxResults=10,  # Récupère plusieurs résultats
        type='video'
    ).execute()

    videos = search_response.get('items', [])
    if not videos:
        return None
    
    # Choisis une vidéo au hasard parmi les résultats
    video = random.choice(videos)
    return video['id']['videoId']

@app.route('/', methods=['GET', 'POST'])
def index():
    audio_url = None
    if request.method == 'POST':
        keyword = request.form['keyword']
        video_id = search_youtube(keyword)
        if video_id:
            audio_url = f"/audio?video_id={video_id}"
    return render_template_string(TEMPLATE, audio_url=audio_url)

@app.route('/audio')
def audio():
    video_id = request.args.get('video_id')
    if not video_id:
        return "No video found", 404

    url = f"https://www.youtube.com/watch?v={video_id}"

    command = [
        'yt-dlp', '-o', '-', '-f', 'bestaudio', url
    ]

    ffmpeg_command = [
        'ffmpeg', '-i', 'pipe:0', '-vn', '-f', 'mp3', 'pipe:1'
    ]

    process1 = subprocess.Popen(command, stdout=subprocess.PIPE)
    process2 = subprocess.Popen(ffmpeg_command, stdin=process1.stdout, stdout=subprocess.PIPE)

    def generate():
        for chunk in iter(lambda: process2.stdout.read(1024), b''):
            yield chunk

    return Response(generate(), content_type='audio/mpeg')

if __name__ == "__main__":
    app.run(debug=True)
