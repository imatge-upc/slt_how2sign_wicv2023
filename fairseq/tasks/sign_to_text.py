# This code is a modification of the "speech_to_text" task, which is licensed
# under the MIT license found in the LICENSE file in the root directory.


import logging
from pathlib import Path, os
from typing import Optional
from argparse import Namespace
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import MISSING, II

from fairseq import metrics, utils
from fairseq.data import AddTargetDataset, Dictionary
#from fairseq.data.encoders.sentencepiece_bpe import SentencepieceConfig
from fairseq.data.sign_language import SignFeatsType, SignFeatsDataset, NormType
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.dataclass.configs import GenerationConfig
from fairseq.scoring import build_scorer
from fairseq.scoring.wer import WerScorerConfig
from fairseq.scoring.bleu import SacrebleuConfig
from fairseq.scoring.chrf import ChrFScorerConfig
from fairseq.tasks import FairseqTask, register_task
from sacremoses import MosesDetokenizer
import truecase

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class SignToTextConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    max_source_positions: Optional[int] = field(
        default=750, metadata={"help": "max number of tokens in the source sequence"}
    )
    min_source_positions: Optional[int] = field(
        default=25, metadata={"help": "min number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=512, metadata={"help": "max number of tokens in the target sequence"}
    )
    normalization: ChoiceEnum([x.name for x in NormType]) = field(
        default=NormType.body.name,
        metadata={"help": "select the type of normalization to apply"},
    )
    data_augmentation: bool = field(
        default=False,
        metadata={"help": "set True to apply data_augmentation to every sample"},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "set True to shuffle the dataset between epochs"},
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = field(
        default="i3d",
        metadata={
            "help": "type of features for the sign input data: mediapipe/i3d/CNN2d/openpose (default: i3d). "
        },
    )
    # Reporting metrics during training
    eval_wer: bool = field(
        default=False,
        metadata={"help": "compute WER on the validation set"}
    )
    eval_wer_config: WerScorerConfig = field(
        default_factory=lambda: WerScorerConfig("wer"),
        metadata={"help": "WER scoring configuration"},
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "compute SacreBLEU on the validation set"}
    )
    pre_tokenizer: str = field(
        default='moses',
        metadata={"help": "detokenization method for BLEU scoring"}
    )
    eval_bleu_config: SacrebleuConfig = field(
        default_factory=lambda: SacrebleuConfig("sacrebleu"),
        metadata={"help": "SacreBLEU scoring configuration"},
    )
    eval_chrf: bool = field(
        default=False,
        metadata={"help": "compute ChRF on the validation set"}
    )
    eval_chrf_config: ChrFScorerConfig = field(
        default_factory=lambda: ChrFScorerConfig("chrf"),
        metadata={"help": "Chrf scoring configuration"},
    )
    eval_reducedBLEU: bool = field(
        default=False,
        metadata={"help": "compute reducedBLEU on the validation set"}
    )
    eval_reducedBLEU_config: SacrebleuConfig = field(
        default_factory=lambda: SacrebleuConfig("sacrebleu"), 
        metadata={"help": "SacreBLEU scoring configuration"},
    )
    eval_reducedchrf: bool = field(
        default=False,
        metadata={"help": "compute reduced ChRF on the validation set"}
    )
    eval_reducedchrf_config: ChrFScorerConfig = field(
        default_factory=lambda: ChrFScorerConfig("chrf"),
        metadata={"help": "Reduced Chrf scoring configuration"},
    )
    eval_gen_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "generaton config for evaluating during training"},
    )
    eval_print_samples: bool = field(
        default=False,
        metadata={"help": "print sample generations during validation"}
    )

    # Inherit from other configs
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    bpe_sentencepiece_model: str = II("bpe.sentencepiece_model")


@register_task("sign_to_text", dataclass=SignToTextConfig)
class SignToTextTask(FairseqTask):
    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        
        self.moses_tokenizer = self.build_tokenizer( #It uses sacremoses inside, similar to what they do here https://github.com/facebookresearch/fairseq/blob/b4001184f49ed0e20d619b54bb3d43088fabf990/fairseq/data/audio/speech_to_text_dataset.py#L692
            Namespace(
                tokenizer=self.cfg.pre_tokenizer
            )
        )
        self.bpe_tokenizer = self.build_bpe(
            Namespace(
                bpe='sentencepiece',
                sentencepiece_model=self.cfg.bpe_sentencepiece_model
            )
        )
        self.scorers = []
        if self.cfg.eval_wer:
            self.scorers.append(
                build_scorer(cfg.eval_wer_config, self.tgt_dict)
            )
        if self.cfg.eval_bleu:
            self.scorers.append(
                build_scorer(cfg.eval_bleu_config, self.tgt_dict)
            )
            if cfg.pre_tokenizer == 'moses':
                self.moses_detok = MosesDetokenizer(lang='en')
        if self.cfg.eval_chrf:
            self.scorers.append(
                build_scorer(cfg.eval_chrf_config, self.tgt_dict)
            )
        if self.cfg.eval_reducedBLEU:
            self.scorers.append(
                build_scorer(cfg.eval_reducedBLEU_config, self.tgt_dict)
            )
            self.scorers[-1].cfg._name = "reducedBLEU"
            if cfg.pre_tokenizer == 'moses':
                self.moses_detok = MosesDetokenizer(lang='en')
        if self.cfg.eval_reducedchrf:
            self.scorers.append(
                build_scorer(cfg.eval_reducedchrf_config, self.tgt_dict)
            )
            self.scorers[-1].cfg._name = "reducedchrf"
            if cfg.pre_tokenizer == 'moses':
                self.moses_detok = MosesDetokenizer(lang='en')
            

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        dict_path = Path(cfg.bpe_sentencepiece_model).with_suffix('.txt')
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({dict_path}): " f"{len(tgt_dict):,}"
        )

        if getattr(cfg, "train_subset", None) is not None:
            if not all("train" in s for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be containe the word "train".')

        if cfg.eval_wer:
            if cfg.eval_wer_config.wer_tokenizer == "none":
                logger.warning(
                    "You are not using any tokenizer for WER scoring. Using '13a' is recommended."
                )
            if not cfg.eval_wer_config.wer_lowercase:
                logger.warning(
                    "You are not lowercasing before WER scoring."
                )
            if not cfg.eval_wer_config.wer_remove_punct:
                logger.warning(
                    "You are not removing punctuation before WER scoring."
                )

        return cls(cfg, tgt_dict)

    def load_dataset(
        self,
        split,
        combine=False,
        task_cfg: FairseqDataclass = None,
        **kwargs
    ):
        is_train_split = "train" in split

        root_dir = Path(self.cfg.data)
        assert root_dir.is_dir(), f"{root_dir} does not exist"

        #Depending on the features type, we load the corresponding manifest file
        manifest_file = root_dir / f"{split}.tsv"
        
        self.datasets[split] = SignFeatsDataset.from_manifest_file(
            manifest_file=manifest_file,
            normalization=self.cfg.normalization,
            data_augmentation=(self.cfg.data_augmentation and is_train_split),
            min_sample_size=self.cfg.min_source_positions,
            max_sample_size=self.cfg.max_source_positions,
            shuffle=self.cfg.shuffle_dataset,
        )

        if is_train_split:
            self.datasets[split] = SignFeatsDataset.from_manifest_file(
                manifest_file=manifest_file,
                normalization=self.cfg.normalization,
                data_augmentation=(self.cfg.data_augmentation and is_train_split),
                min_sample_size=self.cfg.min_source_positions,
                max_sample_size=self.cfg.max_source_positions,
                shuffle=self.cfg.shuffle_dataset,
            )
            self.datasets[split].filter_by_length(
                self.cfg.min_source_positions,
                self.cfg.max_source_positions,
            )
        
        data = pd.read_csv(manifest_file, sep="\t")
        text_compressor = TextCompressor(level=self.cfg.text_compression_level)

        data['translation'] = data['translation'].fillna('')
        labels = [
            text_compressor.compress(row['translation'])
            for i, row in data.iterrows()
            if row['id'] not in self.datasets[split].skipped_ids
        ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"supposed to skip ({len(self.datasets[split].skipped_ids)}) ids"
            f"({len(self.datasets[split])}) do not match"
        )

        def process_label_fn(label):
            if self.cfg.pre_tokenizer == 'moses':
                label = label.lower()
            return self.target_dictionary.encode_line(
                self.bpe_tokenizer.encode(label), append_eos=False, add_if_not_exist=False
            )

        def label_len_fn(label):
            return len(self.bpe_tokenizer.encode(label))
        
        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label_fn,
            label_len_fn=label_len_fn,
            add_to_input=True,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        self.sequence_generator = self.build_generator(
            [model],
            self.cfg.eval_gen_config
        )

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        blacklisted_words = 'examples/sign_language/scripts/blacklisted_words.txt'
        path_to_tasks = os.path.dirname(os.path.realpath(__file__)).split("/fairseq/tasks")[0]
        blacklisted_words_path = os.path.join(path_to_tasks, blacklisted_words)
        with open(blacklisted_words_path, 'r') as file:
            blacklisted_words = file.read().split("\n")

        def decode(toks):
            if hasattr(self.sequence_generator, "symbols_to_strip_from_output"):
                to_ignore = self.sequence_generator.symbols_to_strip_from_output
            else:
                to_ignore = {self.sequence_generator.eos}

            s = self.tgt_dict.string(
                toks.int().cpu(),
                escape_unk=True,
                extra_symbols_to_ignore=to_ignore
            )
            if self.bpe_tokenizer:
                s = self.bpe_tokenizer.decode(s)
            if self.moses_detok:
                s = truecase.get_true_case(s)
            return s

        if len(self.scorers) > 0:
            gen_out = self.inference_step(self.sequence_generator, [model], sample, prefix_tokens=None)
            for i in range(len(gen_out)):
                pred_tok = gen_out[i][0]["tokens"].int().cpu()
                ref = self.datasets[self.cfg.valid_subset].get_label(sample["id"][i])
                pred = decode(pred_tok)
                for s in self.scorers:
                    if s.cfg._name == 'reducedBLEU' or s.cfg._name == 'reducedchrf':
                        no_blacklisted_pred = [word for word in pred.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
                        no_blacklisted_ref = [word for word in ref.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
                        
                        if len(no_blacklisted_pred) == 0:#If the list is empty, we add a space to avoid errors
                            no_blacklisted_pred.append(' ')
                        if len(no_blacklisted_ref) == 0:
                            no_blacklisted_ref.append(' ')
                            
                        s.add_string(' '.join(no_blacklisted_ref), ' '.join(no_blacklisted_pred))
                    else:
                        s.add_string(ref, pred)

                if self.cfg.eval_print_samples:
                    logger.info("Validation example:")
                    logger.info("H-{} {}".format(sample["id"][i], pred))
                    logger.info("T-{} {}".format(sample["id"][i], ref))
        for s in self.scorers:
            if s.cfg._name == 'wer':
                logging_output["_wer_distance"] = s.distance
                logging_output["_wer_ref_len"] = s.ref_length
            elif s.cfg._name == 'sacrebleu':
                sacrebleu_out = s._score()
                logging_output["_bleu_sys_len"] = sacrebleu_out.sys_len
                logging_output["_bleu_ref_len"] = sacrebleu_out.ref_len
                # we split counts into separate entries so that they can be summed efficiently across workers using fast-stat-sync
                assert len(sacrebleu_out.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = sacrebleu_out.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = sacrebleu_out.totals[i]
            elif s.cfg._name == 'chrf':
                logging_output["chrf"] = s.score()
            elif s.cfg._name == 'reducedBLEU':
                reduced_sacrebleu_out = s._score()
                logging_output["_reduced_bleu_sys_len"] = reduced_sacrebleu_out.sys_len
                logging_output["_reduced_bleu_ref_len"] = reduced_sacrebleu_out.ref_len
                assert len(reduced_sacrebleu_out.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_reduced_bleu_counts_" + str(i)] = reduced_sacrebleu_out.counts[i]
                    logging_output["_reduced_bleu_totals_" + str(i)] = reduced_sacrebleu_out.totals[i]
            elif s.cfg._name == 'reducedchrf':
                logging_output["reducedchrf"] = s.score() #TODO: fix this, I think it is empty. Finnd out why
            else:
                raise NotImplemented()

            if utils.safe_hasattr(s, "reset"):
                s.reset()
            else:
                s.ref = []
                s.pred = []

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        #TODO: Check how to only do this for validation
        for s in self.scorers:
            if s.cfg._name == 'wer':
                if sum_logs("_wer_ref_len") > 0:
                    metrics.log_scalar("_wer_distance", sum_logs("_wer_distance"))
                    metrics.log_scalar("_wer_ref_len", sum_logs("_wer_ref_len"))

                    def compute_wer(meters):
                        import torch
                        ref_len = meters["_wer_ref_len"].sum
                        wer = meters["_wer_distance"].sum / ref_len
                        if torch.is_tensor(wer):
                            wer = wer.cpu().item()
                        return round(100 * wer, 2)

                    metrics.log_derived("wer", compute_wer)

            elif s.cfg._name == 'sacrebleu':
                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs("_bleu_counts_" + str(i)))
                    totals.append(sum_logs("_bleu_totals_" + str(i)))
                if max(totals) > 0:
                    # log counts as numpy arrays -- log_scalar will sum them correctly
                    metrics.log_scalar("_bleu_counts", np.array(counts))
                    metrics.log_scalar("_bleu_totals", np.array(totals))
                    metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                    metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                    def compute_bleu(meters):
                        import inspect
                        import torch
                        try:
                            from sacrebleu.metrics import BLEU
                            comp_bleu = BLEU.compute_bleu
                        except ImportError:
                            # compatibility API for sacrebleu 1.x
                            import sacrebleu

                            comp_bleu = sacrebleu.compute_bleu

                        fn_sig = inspect.getfullargspec(comp_bleu)[0]
                        if "smooth_method" in fn_sig:
                            smooth = {"smooth_method": "exp"}
                        else:
                            smooth = {"smooth": "exp"}
                        bleu = comp_bleu(
                            correct=meters["_bleu_counts"].sum,
                            total=meters["_bleu_totals"].sum,
                            sys_len=meters["_bleu_sys_len"].sum if torch.is_tensor(meters["_bleu_sys_len"].sum) == False else meters["_bleu_sys_len"].sum.long().item(),
                            ref_len=meters["_bleu_ref_len"].sum if torch.is_tensor(meters["_bleu_ref_len"].sum) == False else meters["_bleu_ref_len"].sum.long().item(),
                            **smooth,
                        )
                        return round(bleu.score, 2)
                    metrics.log_derived("sacrebleu", compute_bleu)
            
            elif s.cfg._name == 'chrf':
                metrics.log_scalar("chrf", sum_logs("chrf"))
                def compute_chrf(meters):
                    import torch
                    chrf = meters["chrf"]
                    if torch.is_tensor(chrf):
                        chrf = chrf.cpu().item()
                    return chrf
                metrics.log_derived("chrf", compute_chrf)
            
            elif s.cfg._name == 'reducedBLEU':
                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs("_reduced_bleu_counts_" + str(i)))
                    totals.append(sum_logs("_reduced_bleu_totals_" + str(i)))
                if max(totals) > 0:
                    metrics.log_scalar("_reduced_bleu_counts", np.array(counts))
                    metrics.log_scalar("_reduced_bleu_totals", np.array(totals))
                    metrics.log_scalar("_reduced_bleu_sys_len", sum_logs("_reduced_bleu_sys_len"))
                    metrics.log_scalar("_reduced_bleu_ref_len", sum_logs("_reduced_bleu_ref_len"))

                    def compute_reduced_bleu(meters):
                        import inspect
                        import torch

                        try:
                            from sacrebleu.metrics import BLEU
                            comp_bleu = BLEU.compute_bleu
                        except ImportError:
                            # compatibility API for sacrebleu 1.x
                            import sacrebleu
                            comp_bleu = sacrebleu.compute_bleu

                        fn_sig = inspect.getfullargspec(comp_bleu)[0]
                        if "smooth_method" in fn_sig:
                            smooth = {"smooth_method": "exp"}
                        else:
                            smooth = {"smooth": "exp"}
                        bleu = comp_bleu(
                            correct=meters["_reduced_bleu_counts"].sum,
                            total=meters["_reduced_bleu_totals"].sum,
                            sys_len=meters["_reduced_bleu_sys_len"].sum if torch.is_tensor(meters["_reduced_bleu_sys_len"].sum) == False else meters["_reduced_bleu_sys_len"].sum.long().item(),
                            ref_len=meters["_reduced_bleu_ref_len"].sum if torch.is_tensor(meters["_reduced_bleu_ref_len"].sum) == False else meters["_reduced_bleu_ref_len"].sum.long().item(),
                            **smooth,
                        )
                        aux_correct = meters["_reduced_bleu_counts"].sum
                        aux_total = meters["_reduced_bleu_totals"].sum
                        aux_sys_len = meters["_reduced_bleu_sys_len"].sum
                        aux_ref_len = meters["_reduced_bleu_ref_len"].sum
                        return round(bleu.score, 2)
                    metrics.log_derived("reduced_sacrebleu", compute_reduced_bleu)
            elif s.cfg._name == 'reducedchrf':
                metrics.log_scalar("reducedchrf", sum_logs("reducedchrf"))
                def compute_chrf(meters):
                    import torch
                    chrf = meters["reducedchrf"]
                    if torch.is_tensor(chrf):
                        chrf = chrf.cpu().item()
                    return chrf
                metrics.log_derived("reducedchrf", compute_chrf)
            else:
                raise NotImplemented()

    def get_dummy_sample(self):
        subset = self.cfg.train_subset
        subset = subset[0] if isinstance(subset, list) else subset
        self.load_dataset(subset)
        sample = self.datasets[subset][0]
        return sample
