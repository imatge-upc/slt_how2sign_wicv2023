# Script for analizing outputs produced by the models.

import math
from typing import Tuple
from typing import Union
from typing import List
from copy import deepcopy
import argparse
import ast

import pandas as pd
from sympy import ShapeError
import numpy as np

from sklearn import manifold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import  classification_report
from sklearn.calibration import calibration_curve, CalibrationDisplay

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

import torch
import torchvision
import torchvision.transforms.functional as F

from PIL import ImageFont, ImageDraw, ImageOps

# from transformer_contributions_nmt.wrappers.transformer_wrapper import FairseqTransformerHub


tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def load_data_dict(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(file_path, map_location=torch.device('cpu'))
    if type(data) == type(dict()):
        return data  # ['embeddings'], data['targets'], data['preds'], data['att_time'], ['att_inputs']
    raise TypeError(f'Expected data container to be of type `{type(dict)}` but got `{type(data)}` instead.')


def plot_precision_recall(
    targets_binary: List[int],
    preds_binary: List[int],
    model: str,
    data_type: str,
    split: str,
    average: str,
    ) -> None:

    precision = dict()
    recall = dict()
    average_precision = dict()
    precision[average], recall[average], _ = precision_recall_curve(
        targets_binary.ravel(), preds_binary.ravel()
    )
    average_precision[average] = average_precision_score(targets_binary, preds_binary, average=average)

    display = PrecisionRecallDisplay(
        recall=recall[average],
        precision=precision[average],
        average_precision=average_precision[average],
    )
    display.plot()
    _ = display.ax_.set_title(f"{average}-average; {model} - {data_type} - {split}")
    plt.savefig(f'./outputs/{average}-average_precision_recall_{model}_{data_type}_{split}.png')
    plt.close()

    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(targets_binary[:, i], preds_binary[:, i])
        average_precision[i] = average_precision_score(targets_binary[:, i], preds_binary[:, i])

    _, ax = plt.subplots(figsize=(8, 8))

    for i, color in zip(range(10), tab10):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"class {i}", color=color)
    display = PrecisionRecallDisplay(
        recall=recall[average],
        precision=precision[average],
        average_precision=average_precision[average],
    )
    display.plot(ax=ax, name=f"{average}-average precision-recall", color="gold")

    handles, labels = display.ax_.get_legend_handles_labels()
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f'{model} - {data_type} - {split}')
    plt.savefig(f'./outputs/{average}-average_precision_recall_multiclass_{model}_{data_type}_{split}.png')
    plt.close()


def calibration_plots(
    targets: Union[List[int], torch.Tensor],
    logits: torch.Tensor,
    model: str,
    data_type: str,
    split: str,
    ) -> None:

    print(targets)
    print(logits)

    logits = logits.squeeze()
    # fig = plt.figure(figsize=(12, 19))
    # # gs = GridSpec(10, 2)

    # # ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    # # calibration_displays = {}

    # for categ in range(10):
    #     targets_binary = [1 if categ == int(tgt) else 0 for tgt in targets]
    #     logits_categ = logits[:, categ].squeeze().tolist()
    #     prob_true, prob_pred = calibration_curve(targets_binary, logits_categ, n_bins=10, normalize=True,)
    #     display = CalibrationDisplay(
    #         prob_true,
    #         prob_pred,
    #         logits_categ,
    #     )
    #     display.plot(ax=ax_calibration_curve, name=f"class {categ}", color=tab10[categ],)
    #     calibration_displays[categ] = display

    gs = GridSpec(2, 1)
    fig = plt.figure(figsize=(8, 8))
    ax_calibration_curve = fig.add_subplot(gs[0, 0])
    logits_list = logits.tolist()
    targets_list = []
    for tgt in targets.tolist():
        targets_list.append([1 if categ == int(tgt) else 0 for categ in range(1, 11)])

    targets_list = targets.tolist()
    # logits_list = logits.tolist()

    print(targets_list)
    print(logits_list)
    
    prob_true, prob_pred = calibration_curve(targets_list, logits_list, n_bins=10, normalize=True,)
    display = CalibrationDisplay(
        prob_true,
        prob_pred,
        logits_list,
    )
    display.plot(
        ax=ax_calibration_curve,
        name=f"{model}",
        color=tab10[categ],
    )

    plt.grid()
    plt.title(f'{model} - {data_type} - {split}')

    ax = fig.add_subplot(gs[1, 0])

    ax.hist(
        display.y_prob,
        range=(0, 1),
        bins=10,
        label=categ+1,
        color=tab10[categ],
    )
    ax.set(xlabel="Mean predicted probability", ylabel="Count")

    # # Add histogram
    # grid_positions = [(i, j) for i in range(2, 10) for j in range(0, 2)]
    # for categ in range(10):
    #     row, col = grid_positions[categ]
    #     ax = fig.add_subplot(gs[row, col])

    #     ax.hist(
    #         calibration_displays[categ].y_prob,
    #         range=(0, 1),
    #         bins=10,
    #         label=categ+1,
    #         color=tab10[categ],
    #     )
    #     ax.set(title=categ, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig(f'./outputs/calibration_multiclass_{model}_{data_type}_{split}.png')
    plt.close()


def plot_confusion_matrix(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    model: str,
    data_type: str,
    split: str,
    ) -> None:

    disp = ConfusionMatrixDisplay.from_predictions(targets, preds, cmap=plt.cm.Blues, colorbar=False)
    disp.figure_.suptitle(f'{model} - {data_type} - {split}')

    plt.savefig(f'./outputs/confusion_matrix_{model}_{data_type}_{split}.png')
    plt.close()


def metrics_to_csv(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    model: str,
    data_type: str,
    split: str,
    ) -> None:

    report = classification_report(
        targets,
        preds,
        # labels=[i for i in range(1, 11)],
        # target_names=[i for i in range(1, 11)],
        digits=4,
        output_dict=True,
        zero_division='warn',
    )

    report = pd.DataFrame.from_dict(report, orient='columns').transpose()
    report.to_csv(f'./outputs/metrics_report_{model}_{data_type}_{split}.csv')

    support = report.pop('support')
    report, weighted_avg = report.drop(report.tail(1).index),report.tail(1)
    report, macro_avg = report.drop(report.tail(1).index),report.tail(1)
    report, accuracy = report.drop(report.tail(1).index),report.tail(1)

    report = report.append(weighted_avg)
    report = report.append(macro_avg)

    accuracy = accuracy.iloc[0,0]

    ax = report.plot.bar(
        rot=0,
        width=0.7,
        edgecolor='white',
        linewidth=1.5,
        color=["#ff7f0e", "#bcbd22", "#8c564b"],
        figsize=(11, 5),
    )
    ax.axes.set_xlim(-0.5,11.5)
    leg1 = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fancybox=True,
        shadow=True
    )

    leg2 = ax.legend(
        [f"accuracy = " + "{:.2f}".format(accuracy*100)],
        handles=[
            Line2D(
                [0], [0], marker='o', color='w', label=f"accuracy = " + "{:.2f} %".format(accuracy*100),
                markerfacecolor='g', markersize=0)
        ],
        loc='upper center',
        bbox_to_anchor=(0.85, 1.065),
        borderaxespad=0,
        fontsize='x-large',
        frameon=False,
    )
    ax.add_artist(leg1)

    plt.xticks([i for i in range(12)], [i for i in range(1, 11)] + ['w_avg', 'macro_avg'])
    plt.savefig(f'./outputs/metrics_barchart_{model}_{data_type}_{split}.png')
    plt.close()

    ax = support.iloc[0:10].plot.bar(rot=0, width=0.7, edgecolor='white', linewidth=1, color=tab10)
    ax.set_title(f"Samples per class in {split} set")
    plt.xticks([i for i in range(10)], [i for i in range(1, 11)])
    plt.savefig(f'./outputs/metrics_support_{model}_{data_type}_{split}.png')
    plt.close()


def analysis_of_errors(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    logits: Union[List[float], torch.Tensor],
    labels: List[str],
    model: str,
    data_type: str,
    split: str,
    ) -> None:
    from sklearn.metrics import precision_recall_fscore_support as score

    # targets = [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,1,2,1,1,4,1]
    # preds   = [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,1,2,1,1,4,5]
    # logits = []
    # labels = [f'class_{i}' for i in range(10)]
    # for i in range(21):
    #     array = np.random.normal(0.5, 10, 10)
    #     array /= np.sum(array)
    #     logits.append(array.tolist())
    # logits = torch.tensor(logits)

    from sklearn.preprocessing import label_binarize
    # Use label_binarize to fit into a multilabel setting
    targets_binary = label_binarize(targets, classes=[i for i in range(10)])
    preds_binary = label_binarize(preds, classes=[i for i in range(10)])
    # TODO: make sure that targets and preds are binarized the same way

    for average in ['micro', 'macro']:
        plot_precision_recall(
            targets_binary=targets_binary,
            preds_binary=preds_binary,
            model=model,
            data_type=data_type,
            split=split,
            average=average,
        )

    plot_confusion_matrix(
        targets=targets,
        preds=preds,
        model=model,
        data_type=data_type,
        split=split,
    )

    metrics_to_csv(
        targets=targets,
        preds=preds,
        model=model,
        data_type=data_type,
        split=split,
    )


def plot_att_time(
    att_time: torch.Tensor,
    video_id: str,
    model: str,
    data_type: str,
    ) -> None:

    att_time = att_time[0]  # TODO: remove this

    print(att_time)
    if att_time.shape[0] == 1 and len(att_time.shape) == 2:
        att_time -= att_time.min(1, keepdim=True)[0]
        att_time /= att_time.max(1, keepdim=True)[0]
    elif len(att_time.shape) == 1:
        att_time -= att_time.min(0, keepdim=True)[0]
        att_time /= att_time.max(0, keepdim=True)[0]
    else:
        raise ShapeError('Please pass in a tensor corresponding to just one sample')
    if model == 'transformer':
        raise TypeError('Temporal attention visualization is not implemented for Transformer, only for PerceiverIO and LSTM.')

    att_time = att_time.tolist()
    print(f'len(att_time) = {len(att_time)}')

    text_timestamps = pd.read_csv(
        (
            '/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/'
            'train_csv_frames_redone_2.csv'
        ),
        sep = '\t'
    )
    text_timestamps = text_timestamps[text_timestamps['VIDEO_ID'] == video_id]
    text_timestamps = text_timestamps.sort_values(by=['START_FRAME'])

    word_timestamp_list = []
    if data_type == 'text':  # TODO: retrieve tokenized text to visualize it
        pass
    else:
        for index, row in text_timestamps.iterrows():
            print(f'index = {index}')
            text = row['SENTENCE']
            words = text.split(' ')
            l = len(words)
            start_frame = row['START_FRAME'] - 8 if data_type == 'i3d' else row['START_FRAME']
            end_frame = row['END_FRAME'] - 8 if data_type == 'i3d' else row['END_FRAME']
            delta_t = end_frame - start_frame
            delta_w = delta_t / l
            word_frame = [(math.floor(i * delta_w), word) for i, word in enumerate(words)]

            expand = max(1, math.floor((delta_t / 100) ** 1.2))
            print(f'expand = {expand}')

            x = [i for i in range(0, delta_t, 1)]
            y = [1 for _ in range(0, delta_t, 1)]
            print(f'(start_frame, end_frame) = {(start_frame, end_frame)}')
            c = att_time[start_frame:end_frame]

            # viz. weights
            dpi = 120
            fig, ax = plt.subplots(figsize=(1000 / dpi * expand, 1000 / dpi * expand), dpi=dpi)
            ax.axis('off')

            ax.scatter(x, y, c=c, alpha=0.8, marker='s', s=1)

            for i, word in word_frame:
                print(f'i = {i}')
                ax.annotate(word, (x[i], y[i]))

            plt.set_cmap('YlOrBr')
            plt.savefig(f'./outputs/attention_{model}_{data_type}_{start_frame}_{end_frame}.png')
            plt.close()

    pass


def viz_att(
    att_time: torch.Tensor,
    video_path: str,
    csv_path:str,
    model: str,
    data_type: str,
    ) -> None:
    """
    To visualize attentions on video:
        * LSTM, PerceiverIO, Transformer:
            1. do inference on the video's data with models trained on Calcula
            2. load those att_time weights and pass them in to `viz_att` function
        * TransformerCLS:
            1. do inference on the video's data with transformerCLS model trained on Calcula.
               It will be necessary to hand-code the call to get the ALTI contribution weights
            2. load those ALTI weights and pass them in to `viz_att` function
    """
    print(f'att_time.shape = {att_time.shape}')
    if model == 'transformerCLS':
        att_time_path = (
            f'/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/fairseq-internal/'
            f'examples/SL_topic_detection/outputs/att_time_ALTI_transformerCLS_{data_type}_train.pkl'
        )
        att_time = torch.load(att_time_path)
    else:
        att_time = att_time[0]
        att_time = att_time.squeeze()
    print(f'att_time.shape = {att_time.shape}')

    if att_time.shape[0] == 1 and len(att_time.shape) == 2:
        att_time -= att_time.min(1, keepdim=True)[0]
        att_time /= att_time.max(1, keepdim=True)[0]
        att_time /= att_time.mean(1, keepdim=True)[0]
    elif len(att_time.shape) == 1:
        att_time -= att_time.min()
        att_time /= att_time.max()
        att_time /= att_time.mean()
    else:
        raise ShapeError('Please pass in a tensor corresponding to just one sample')

    # gloss_timestamp = pd.read_csv(csv_path, sep=';')
    text_font = ImageFont.truetype('/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/att_viz/train/Verdana.ttf', 33)

    if data_type == 'spot_align':
        csv_path = '/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/spot_align/train.csv'
        mouthings = pd.read_csv(csv_path)
        mouthings = mouthings[mouthings['VIDEO_ID'] == '-EsVrbRTMU4']
        idx = [5, 9, 19, 28, 35, 45, 63, 75, 80, 86, 101, 105, 107]
        words = list(mouthings['TEXT'])[0]
        w_frames = list(mouthings['MOUTHING_FRAME'])[0]
        words = ast.literal_eval(words)
        w_frames = ast.literal_eval(w_frames)

        words = [words[i] for i in idx]
        w_frames = [w_frames[i] for i in idx]
        for i, w, w_f in zip(idx, words, w_frames):
            init_frame, end_frame = int(w_f - 10), int(w_f + 10)
            print(f'init_frame, end_frame = {init_frame, end_frame}')
            init_time, end_time = init_frame / 25, end_frame / 25
            video = torchvision.io.read_video(
                filename=video_path,
                start_pts=init_time,
                end_pts=end_time,
                pts_unit='sec'
            )
            frame_rate = video[2]['video_fps']
            video = video[0]
            print(f'video.shape = {video.shape}')

            print('%'*25)
            out = []

            # define color mapping for the att weights
            N = 1000
            cmap = cm.get_cmap('plasma', N)

            # for each frame in the video range...
            for t in range(min(end_frame - init_frame, video.shape[0])):
                frame = video[t].permute(2, 0, 1)
                frame = F.to_pil_image(frame)
                image_editable = ImageDraw.Draw(frame)
                image_editable.text(
                    (15,15),  # (0, 0): upper left
                    w,
                    (0, 0, 0),  # RGB
                    font=text_font,
                )

                # add sourrounding frame with weight's coloring to each of the frames in video
                color_att = int(N * att_time[i])
                colors = tuple(int(255 * c) for c in cmap(color_att)[0:3])
                frame = ImageOps.expand(frame, border=66, fill=colors)  # (0,0,0))
                frame = F.pil_to_tensor(frame).permute(1, 2, 0)
                out.append(frame)

            # store to .mp4
            out = torch.stack(out)
            out_path = '/'.join(video_path.split('/')[:-1]) + '/' + f'{data_type}_{model}_viz_att_{init_frame}' + video_path.split('/')[-1]
            print(out_path)
            torchvision.io.write_video(
                filename=out_path,
                video_array=out,
                fps=frame_rate,
                video_codec='libx264',
            )
    else:
        text_timestamps = pd.read_csv(
            (
                '/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/'
                'train_csv_frames_redone_2.csv'
            ),
            sep = '\t'
        )
        text_timestamps = text_timestamps[text_timestamps['VIDEO_ID'] == '-EsVrbRTMU4']
        text_timestamps = text_timestamps.sort_values(by=['START_FRAME'])

        if (
                data_type in ['keypoints', 'mediapipe_keypoints', 'mediapipe_rotational'] and  # TODO: all non-textual feats should have been downsampled!
                model == 'transformer'
            ):
                print(f'1 att_time.shape = {att_time.shape}')
                att_time = att_time.repeat_interleave(9)
                print(f'2 att_time.shape = {att_time.shape}')
        elif (
                data_type in ['keypoints', 'mediapipe_keypoints', 'rotational', 'mediapipe_rotational', 'i3d',] and
                model == 'transformerCLS'
            ):
                print(f'1 att_time.shape = {att_time.shape}')
                att_time = torch.from_numpy(att_time).repeat_interleave(9)
                print(f'2 att_time.shape = {att_time.shape}')

        for i, row in text_timestamps.iterrows():
            init_time, end_time = int(row['START_REALIGNED']), int(row['END_REALIGNED'])
            init_frame, end_frame = int(row['START_FRAME']), int(row['END_FRAME'])
            if init_frame > 1200:
                break
            if data_type == 'i3d':
                init_frame, end_frame = max(0, init_frame - 8), max(0, end_frame - 8)
                if end_frame == 0:
                    end_frame += 8
            print(f'init_frame, end_frame = {init_frame, end_frame}')

            ### add subtitle to time range (init_time, end_time) ###
            video = torchvision.io.read_video(
                filename=video_path,
                start_pts=init_time,
                end_pts=end_time,
                pts_unit='sec'
            )
            frame_rate = video[2]['video_fps']
            video = video[0]
            print(f'video.shape = {video.shape}')

            print('%'*25)
            out = []

            # define color mapping for the att weights
            N = 1000
            cmap = cm.get_cmap('plasma', N)

            # add subtitles...
            text = row['SENTENCE']
            print(f'len(text) = {len(text)}')

            ##
            words = text.split()
            new_text = ""
            word_count = 0
            for word in words:
                new_text += word + " "
                word_count += 1
                if word_count == 11 or "." in word:
                    new_text += "\n"
                    word_count = 0
            ##
            text = new_text
            # for each frame in the video range...
            for t in range(min(end_frame - init_frame, video.shape[0])):
                # print(f't = {t}')
                frame = video[t].permute(2, 0, 1)
                # print(frame.shape)
                frame = F.to_pil_image(frame)
                image_editable = ImageDraw.Draw(frame)
                image_editable.text(
                    (15,15),  # (0, 0): upper left
                    text,
                    (0, 0, 0),  # RGB
                    font=text_font,
                )

                # add sourrounding frame with att weight's coloring to each of the frames in video
                color_att = int(N * att_time[init_frame + t])
                colors = tuple(int(255 * c) for c in cmap(color_att)[0:3])
                frame = ImageOps.expand(frame, border=66, fill=colors)
                frame = F.pil_to_tensor(frame).permute(1, 2, 0)
                out.append(frame)

            # store to .mp4
            out = torch.stack(out)
            out_path = '/'.join(video_path.split('/')[:-1]) + '/' + f'{data_type}_{model}_viz_att_{init_frame}' + video_path.split('/')[-1]
            print(out_path)
            torchvision.io.write_video(
                filename=out_path,
                video_array=out,
                fps=frame_rate,
                video_codec='libx264',
            )


def obtain_tSNE_projection(
    embeddings: Union[torch.Tensor, np.array],
    ) -> np.array:
    # TODO: set a grid for each of the models (3 in total),
    #       with 4 x 3 = 12 subplots each (4 data types, 3 dataset splits)
    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    if len(embeddings.shape) != 2:
        raise RuntimeError(
            (f'Expected input embeddings to be two-dimensional tensor'
            f' but got a `{len(embeddings.shape)}-dimensional tensor instead.`')
        )
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate="auto",
        n_iter=1000,
        random_state=41,
        n_jobs=-1,
    )
    Y = tsne.fit_transform(embeddings)
    return Y


def plot_projection(
    Y: np.array,
    class_labels: Union[List[int], List[str], torch.Tensor],
    labels: List[str],
    model: str,
    data_type: str,
    split: str,
    ) -> None:
    if type(class_labels) == torch.Tensor:
        class_labels = deepcopy(class_labels).tolist()
    # toy example
    # class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # y = [0, 1, 2, 2, 2, 2, 3, 3, 4, 4,  5, 5]

    dpi = 100
    fig, ax = plt.subplots(figsize=(850/dpi,850/dpi), dpi=dpi)

    ax.axis('off')
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=class_labels, cmap='tab10')
    # scatter = plt.scatter(x, y, c=class_labels, cmap='tab10')
    # lgd = ax.legend(handles=scatter.legend_elements()[0], labels=labels, ncol=1, bbox_to_anchor=(1.04,1))
    ax.set_title(f'{model} - {data_type} - {split}')
    # plt.savefig(f'./outputs/tsne_{model}_{data_type}_{split}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(f'./outputs/tsne_{model}_{data_type}_{split}.png', bbox_inches='tight')
    plt.close()


# maps the label's number to its corresponding string e.g. 0 -> "beauty"
def label_num_to_str(
    class_labels: Union[List[int], List[str], torch.Tensor],
    mapping_file: str = '../../../data/How2Sign/categoryName_categoryID.csv',
    ) -> List[str]:
    if type(class_labels) == torch.Tensor:
        class_labels = deepcopy(class_labels).tolist()

    mapping = pd.read_csv(mapping_file, index_col=0, squeeze=True, sep=',').to_dict()
    assert len(mapping.keys()) == 10
    return mapping


def obtain_labels(targets):
    mapping = label_num_to_str(targets)
    labels = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
    return labels


def main(args):
    MODELS = ['perceiverIO', 'lstm', 'transformer', 'transformerCLS']
    DATA_TYPE = ['keypoints', 'mediapipe_keypoints', 'rotational', 'mediapipe_rotational', 'text', 'i3d', 'spot_align', 'mouthings']
    SPLIT = ['test', 'val', 'train']

    for model in MODELS:
        for data_type in DATA_TYPE:
            for split in SPLIT:
                print(f'Analyzing outputs from model = {model}, data type = {data_type}, split = {split}...', flush=True)
                print(flush=True)

                data = load_data_dict(f'./outputs/inference_{model}_{data_type}_{split}.pt')

                # if model == 'transformerCLS':
                #     data = {'att_time': np.array([[-99999, -99], [-99999, -99]])}
                # else:
                #     data = load_data_dict(f'./outputs/inference_{model}_{data_type}_{split}.pt')
                targets = data['targets'] + 1
                preds = data['preds'] + 1
                logits = data['logits']
                labels = obtain_labels(data['targets'])

                Y = obtain_tSNE_projection(data['embeddings'])
                print(f'Plotting projections; model = {model}, data type = {data_type}, split = {split}...', flush=True)
                plot_projection(Y, targets, obtain_labels(data['targets']), model, data_type, split)

                print(f'analysis_of_errors; model = {model}, data type = {data_type}, split = {split}...', flush=True)
                analysis_of_errors(
                    targets=targets,
                    preds=preds,
                    logits=logits,
                    labels=labels,
                    model=model,
                    data_type=data_type,
                    split=split,
                )

                # data_dir = f'../../../../../../data/How2Sign/{data_type}'
                # plot_att_time(
                #     att_time=data['att_time'][0],
                #     video_id='eSzXQQUgH1A',
                #     model=model,
                #     data_type=data_type,
                # )

                # viz_att(
                #     att_time=data['att_time'],  # TODO: select the data corresponding to the -EsVrbRTMU4 video
                #     video_path='/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/att_viz/train/-EsVrbRTMU4-8-rgb_front.mp4',
                #     # csv_path='/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/video/train/-EsVrbRTMU4-8-rgb_front_gloss_timesteps.csv',
                #     csv_path='/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/train_csv_frames_redone_2.csv',
                #     model=model,  # 'lstm',
                #     data_type=data_type,
                # )

                print(flush=True)
                print(f'Analyzed outputs for model = {model}, data type = {data_type}, split = {split}', flush=True)
                print(flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
