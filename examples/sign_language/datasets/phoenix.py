import pandas as pd
from pathlib import Path
from pandarallel import pandarallel

from pose_format.pose import Pose

from . import SignLanguageDataset, register_dataset


@register_dataset('phoenix')
class Phoenix(SignLanguageDataset):
    SPLITS = ['train', 'train-complex-annotation', 'dev', 'test']
    SIGNS_TYPES = ['mediapipe'] # TODO: Add i3d and 2dcnn
    SIGNS_LANGS = ['dgs']
    TRANSLATION_LANGS = ['de']

    def __init__(self, base_dir: Path, split: str, signs_type: str,
                 signs_lang: str, translation_lang: str) -> None:
        super().__init__(base_dir, split, signs_type, signs_lang, translation_lang)

        csv_orginal = self.base_dir / \
                      "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual" / \
                      f"PHOENIX-2014-T.{split}.corpus.csv"
        df_original = pd.read_csv(csv_orginal.as_posix(), sep='|')

        pandarallel.initialize(progress_bar=False, verbose=1)

        self.data = pd.DataFrame(columns=self.MANIFEST_COLUMNS)

        self.data['id'] = df_original['name']

        split_ = split.split('-')[0]
        get_signs_file = lambda x: (
            self.base_dir / "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features" / \
            signs_type / split_ / f"{x['id']}.pose"
        ).as_posix()
        self.data['signs_file'] = self.data.parallel_apply(get_signs_file, axis=1
        )

        def get_pose_length(pose_file: str) -> int:
            with open(pose_file, "rb") as f:
                p = Pose.read(f.read())
            return p.body.data.shape[0]

        self.data['signs_offset'] = 0
        self.data['signs_length'] = self.data.parallel_apply(
            lambda x: get_pose_length(x['signs_file']), axis=1
        )

        self.data['signs_type'] = signs_type
        self.data['signs_lang'] = signs_lang

        self.data['translation'] = df_original['translation']
        self.data['translation_lang'] = translation_lang

        self.data['glosses'] = df_original['orth']
        self.data['signer_id'] = df_original['speaker']
