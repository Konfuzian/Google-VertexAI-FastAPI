from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=DbsAQSIKQXk")
print(yt.title)
print(yt.thumbnail_url)

stream = yt.streams.filter(only_audio=True, audio_codec="opus")[0]
print(stream)
stream.download(output_path='downloads/')
