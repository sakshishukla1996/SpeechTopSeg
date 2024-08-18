import glob
import subprocess
import sys
import yt_dlp


class ConvertPP(yt_dlp.postprocessor.PostProcessor):
    def run(self, info):
        filepath_orig = info["filepath"]
        filepath_wav = subset + "/wav/" + info["id"] + ".wav"
        subprocess.call(
            f"ffmpeg -y -v 0 -i {filepath_orig} -acodec pcm_s16le -ar 16000 -ac 1 -f wav {filepath_wav}",
            shell=True,
        )
        info["filepath"] = filepath_wav
        return [filepath_orig], info


for subset in [
    "akashvani",
    "euronews_de",
    "euronews_en",
    "euronews_es",
    "euronews_fr",
    "euronews_it",
    "euronews_pt",
    "tagesschau",
]:
    video_ids = [x[-16:-5] for x in glob.glob(f"{subset}/json/*.json")]

    ydl = yt_dlp.YoutubeDL(
        {"format": "bestaudio/best", "outtmpl": subset + "/wav/%(id)s.%(ext)s"}
    )
    ydl.add_post_processor(ConvertPP())
    ydl.download(video_ids)
