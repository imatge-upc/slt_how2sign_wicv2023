import torch
import torch.nn.functional as F

from typing import List, Tuple, Optional

from fairseq.data.sign_language import SignFeatsType

def get_num_feats(
    feats_type: SignFeatsType,
    bodyparts: Optional[List[str]] = None,
    feat_dims: Optional[List[int]] = None
) -> int:
    num_feats = {
        SignFeatsType.i3d: 1024,
        SignFeatsType.CNN2d: 1024,
        SignFeatsType.video: (720, 1280),
        SignFeatsType.keypoints: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21
        },
        SignFeatsType.mediapipe_keypoints: {
            'face': 70,
            'upperbody': 8,
            'lowerbody': 16,
            'lefthand': 21,
            'righthand': 21
        },
        SignFeatsType.rotational: 288,
        SignFeatsType.mediapipe_rotational: 288,
        SignFeatsType.text: 256,  # TODO: decide which dim to return, or if this function should be called at all when using text as input
        SignFeatsType.text_albert: 768,
        SignFeatsType.spot_align: 256,
        SignFeatsType.spot_align_albert: 768,
        SignFeatsType.mouthings: 256,
        SignFeatsType.mouthings_albert: 768,

    }
    if (feats_type is SignFeatsType.i3d or
        feats_type is SignFeatsType.CNN2d or
        feats_type is SignFeatsType.video or
        feats_type is SignFeatsType.rotational or
        feats_type is SignFeatsType.mediapipe_rotational or
        feats_type is SignFeatsType.text or
        feats_type is SignFeatsType.text_albert or
        feats_type is SignFeatsType.spot_align or
        feats_type is SignFeatsType.spot_align_albert or
        feats_type is SignFeatsType.mouthings or
        feats_type is SignFeatsType.mouthings_albert
        ):
        return num_feats[feats_type]
    elif feats_type in [SignFeatsType.keypoints, SignFeatsType.mediapipe_keypoints]:
        return sum([num_feats[feats_type][b] for b in bodyparts]) * len(feat_dims)
    else:
        raise AttributeError(f"Feat type selected not supported: {feats_type}")


def select_keypoints_by_bodypart(
        keypoints: torch.Tensor,
        feats_type: SignFeatsType,
        bodyparts: Optional[List[str]] = None,
        datasetType: str = 'How2Sign',
) -> Tuple[torch.Tensor, int]:
    if datasetType == 'Phoenix' or SignFeatsType[feats_type] in [SignFeatsType.mediapipe_keypoints]:  # TODO: make sure that in task the correct value for keypoints_type is passed
        return keypoints.reshape(-1, 50*3).contiguous(), 50

    BODY_IDX = {
        'face': torch.arange(70),           # 0-69
        'upperbody': torch.arange(70,78),   # 70-78
        'lowerbody': torch.arange(78,95),   # 79-94
        'lefthand': torch.arange(95,116),   # 95-115
        'righthand': torch.arange(116,137)  # 116-136
    }

    if bodyparts is None:
        bodyparts = list(BODY_IDX.keys())

    assert len(bodyparts) > 0, "You haven't selected any bodypart!"
    assert all([b in BODY_IDX.keys() for b in bodyparts]), f"You have selected a bodypart that doesn't exist! The options are: {list(BODY_IDX.keys())}"

    selected_idx = torch.cat([BODY_IDX[b] for b in bodyparts])

    keypoints = keypoints.reshape(-1, 137, 4)
    keypoints_selected = keypoints[:, selected_idx]
    keypoints = keypoints_selected.reshape(-1, len(selected_idx) * 4).contiguous()

    return keypoints, len(selected_idx)


def select_keypoints_by_dimension(
        keypoints: torch.Tensor,
        dimensions: List[int],
        feats_type: SignFeatsType,
        datasetType: str = 'How2Sign',
) -> torch.Tensor:
    assert len(dimensions) > 0, "You haven't selected any dimensions!"
    assert all([idx<4 for idx in dimensions]), "You have selected a dimension that doesn't exist! The options are: 0 for x, 1 for y, 2 for z and 3 for confidence score "
    if datasetType == 'Phoenix' or SignFeatsType[feats_type] in [SignFeatsType.mediapipe_keypoints]:  # TODO: make sure that in task the correct value for keypoints_type is passed
        return keypoints.reshape(-1, 50*3).contiguous()

    selected_idx = torch.LongTensor(dimensions)

    n_keypoints = int(keypoints.size(-1) / 4)
    keypoints = keypoints.reshape(-1, n_keypoints, 4)
    keypoints_selected = keypoints[:, :, selected_idx]
    keypoints = keypoints_selected.reshape(-1, n_keypoints * len(selected_idx)).contiguous()  

    return keypoints