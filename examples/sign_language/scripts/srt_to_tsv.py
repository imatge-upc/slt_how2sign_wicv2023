import datetime
import srt
import os
import csv
import argparse

def miliseconds_to_frame_index(miliseconds: int, fps: int) -> int:
    """
    :param miliseconds:
    :param fps:
    :return:
    """
    return int(fps * (miliseconds / 1000))

def convert_srt_time_to_frame(srt_time: datetime.timedelta, fps: int) -> int:
    """
    datetime.timedelta(seconds=4, microseconds=71000)
    :param srt_time:
    :param fps:
    :return:
    """
    seconds, microseconds = srt_time.seconds, srt_time.microseconds
    miliseconds = int((seconds * 1000) + (microseconds / 1000))
    return miliseconds_to_frame_index(miliseconds=miliseconds, fps=fps)

def subtitle_is_usable(subtitle: srt.Subtitle, fps: int) -> bool:
    """
    :param subtitle:
    :param fps:
    :return:
    """
    if subtitle.content.strip() == "":
        print("Skipping empty subtitle: %s" % str(subtitle))
        return False
    start_frame = convert_srt_time_to_frame(subtitle.start, fps=fps)
    end_frame = convert_srt_time_to_frame(subtitle.end, fps=fps)
    if not start_frame < end_frame:
        print("Skipping subtitle where start frame is equal or higher than end frame: %s" % str(subtitle))
        return False
    return True

def get_file_id(filename: str) -> str:
    """
    Examples:
    - srf.2020-03-12.srt
    - focusnews.120.srt
    :param filename:
    :return:
    """
    parts = filename.split(".")
    return parts[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtitle_dir', type=str, default='/home/usuaris/gerard.ion.gallego/datasets/WMT-SLT22/focusnews/subtitles', help='path of the folder where the subtitles are')
    args = parser.parse_args()
    subtitle_dir = args.subtitle_dir

    subtitles=[]
    subtitles_by_id = {}
    fps=25
    num_subtitles_skipped=0
    header = ['VIDEO_ID', 'SENTENCE_ID', 'START_FRAME', 'END_FRAME', 'TEXT']
    with open('/home/usuaris/gerard.ion.gallego/datasets/WMT-SLT22/focusnews/subtitles/training_subtitles.tsv', 'w+', newline='') as f_output:
        tsv_writer = csv.writer(f_output, delimiter='\t')
        tsv_writer.writerow(header)
        for filename in os.listdir(subtitle_dir):
            filepath = os.path.join(subtitle_dir, filename)
            file_id = get_file_id(filename)
            with open(filepath, "r") as f_srt:
                for subtitle in srt.parse(f_srt.read()):
                    # skip if there is no text content or times do not make sense
                    if not subtitle_is_usable(subtitle=subtitle, fps=fps):
                        num_subtitles_skipped += 1
                        continue
                    #EL QUE NECESSITO PEL TSV, però he de make sure que els fps son aquests
                    index = subtitle.index #It starts with 1.
                    start_frame = convert_srt_time_to_frame(subtitle.start, fps=fps)
                    end_frame = convert_srt_time_to_frame(subtitle.end, fps=fps)
                    text = subtitle.content
                    tsv_writer.writerow([file_id, str(index), str(start_frame), str(end_frame), text])

#Save them: in the TSV: it should have:
#VIDEO_NAME, SENTENCE_NAME, START_TIME, START_FRAME, END_TIME, END_FRAME, TEXT 
