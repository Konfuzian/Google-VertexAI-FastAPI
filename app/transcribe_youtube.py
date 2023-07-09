from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(link: str):
    video_id = link.split("=")[-1]
    
    response = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([i["text"] for i in response])
    return text
