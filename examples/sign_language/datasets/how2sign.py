import pandas as pd
from pathlib import Path
from pandarallel import pandarallel

from . import SignLanguageDataset, register_dataset
import re

@register_dataset('how2sign')
class How2Sign(SignLanguageDataset):
    """
    Create a Dataset for How2Sign. Each item is a dictionary with:
    id, file, offset, length, type, signs_lang, translation, translation_lang, topic
    base dir should be:
    """

    TRANSLATION_LANGS = ['en'] # TODO: add 'pt'
    SPLITS = ['train', 'val', 'test']
    SIGNS_TYPES = ['mediapipe', 'i3d'] #TODO: Add 'i3d'
    SIGNS_LANGS = ['asl']

    def __init__(self, base_dir: Path, split: str, signs_type: str, signs_lang: str, translation_lang: str) -> None:
        super().__init__(base_dir, split, signs_type, signs_lang, translation_lang)
        #TODO: add adding_topic_tsv.py, integrate in this script.
        
        if signs_type == 'mediapipe':
            tsv_orginal = self.base_dir / \
                        "How2Sign/metadata" / \
                        f"cvpr23.{signs_type}.{split}.how2sign.tsv"
            df_original = pd.read_csv(tsv_orginal.as_posix(), sep='\t')
            
            pandarallel.initialize(progress_bar=False, verbose=1)

            self.data = pd.DataFrame(columns=self.MANIFEST_COLUMNS)

            self.data['id'] = df_original['SENTENCE_NAME']

            find_groups = r'(^.{11})(.*)-([0-9]+)(.*)'
            print_groups = r'\1-\3\4' #backlash is not allowed inside f-string
            
            get_signs_file = lambda x: (
                self.base_dir / "How2Sign/video_level" / \
                split / "rgb_front/features" / signs_type / f"{re.sub(find_groups, print_groups, x['id'])}.pose"
            ).as_posix()
            
        self.data['signs_file'] = self.data.parallel_apply(get_signs_file, axis=1)
        
        def get_pose_length(pose_file: str) -> int:
            with open(pose_file, "rb") as f:
                p = Pose.read(f.read())
            return p.body.data.shape[0]

        self.data['signs_offset'] = df_original['START_FRAME']
        self.data['signs_length'] = df_original['END_FRAME'] - df_original['START_FRAME'] + 1

        self.data['translation'] = df_original['SENTENCE']
        self.data['topic'] = df_original['TOPIC_ID']

        self.data['signs_type'] = signs_type
        self.data['signs_lang'] = signs_lang
        self.data['translation_lang'] = translation_lang
