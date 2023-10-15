import sys
import threading as t
from concurrent.futures import ProcessPoolExecutor

from utils.audio import Audio
from utils.video import Video_detect_face


def _main(video_flag: bool, audio_flag: bool) -> None:
    audio, video = None, None
    if video_flag:
        video = Video_detect_face()
        # t1 = t.Thread(target=_run)
        t1 = t.Thread(target=video.stream_video)
        with ProcessPoolExecutor(4) as exe:
            exe.submit(t1.start())
    if audio_flag:
        audio = Audio()
        t2 = t.Thread(target=audio.stream_audio, daemon=True)
        t2.start()
    if video:
        t1.join()
    if audio:
        t2.join()
    return


if __name__ == '__main__':
    _main(True, True)
    exit()
