import pandas as pd
from pathlib import Path

from . import SignLanguageDataset, register_dataset


@register_dataset('how2sign')
class How2Sign(SignLanguageDataset):
    """
    Create a Dataset for How2Sign. Each item is a dictionary with:
    id, signs, topic, translation
    """

    LANGUAGES = ['en'] # TODO: add 'pt'
    SPLITS = ['train', 'val', 'test']
    FEATS_TYPES = ['mp', 'op'] #TODO: Add 'i3d'

    def __init__(self, tsv_file: Path, feats_file: Path) -> None:
        super().__init__(tsv_file, feats_file)
        df_in = pd.read_csv(self.tsv_file.as_posix(), sep="\t")
        self.vids_error = []
        for i, row in df_in.iterrows():
            vid_id = row['VIDEO_NAME']
            if vid_id not in self.feats_h5.keys():
                if vid_id not in self.vids_error:
                    print(f"Error with video {vid_id}")
                    self.vids_error.append(vid_id)
                df_in.drop(i, inplace=True)
        df_in.reset_index(drop=True, inplace=True)

        self.data = pd.DataFrame(columns=self.MANIFEST_COLUMNS)
        self.data['id'] = df_in['SENTENCE_NAME']
        self.data['h5_id'] = self.data.apply(lambda x: f"{x['id'][:11]}-{x['id'][12:].split('-', 1)[1]}", axis=1)
        self.data['signs_file'] = self.feats_file.as_posix()
        self.data['offset'] = df_in['START_FRAME']
        self.data['length'] = df_in['END_FRAME'] - df_in['START_FRAME'] + 1
        self.data['translation'] =  df_in['SENTENCE']
