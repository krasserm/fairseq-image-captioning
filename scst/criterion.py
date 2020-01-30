import torch

from fairseq.data import encoders
from fairseq.criterions import FairseqCriterion, register_criterion

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

from scst.generator import SimpleSequenceGenerator


@register_criterion('self_critical_sequence_training')
class SelfCriticalSequenceTrainingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.task = task

        self.generator = SimpleSequenceGenerator(beam=args.scst_beam,
                                                 penalty=args.scst_penalty,
                                                 max_pos=args.max_target_positions,
                                                 eos_index=task.target_dictionary.eos_index)

        # Needed for decoding model output to string
        self.conf_tokenizer = encoders.build_tokenizer(args)
        self.conf_decoder = encoders.build_bpe(args)
        self.captions_dict = task.target_dictionary

        # Tokenizer needed for computing CIDEr scores
        self.tokenizer = PTBTokenizer()
        self.scorer = Cider()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--scst-beam', type=int, default=5,
                            help='beam size')
        parser.add_argument('--scst-penalty', type=float, default=1.0,
                            help='beam search length penalty')
        parser.add_argument('--scst-validation-set-size', type=int, default=0, metavar='N',
                            help='limited size of validation set')

    @property
    def image_ids(self):
        return self.task.dataset('train').img_ds.image_ids

    def decode(self, x):
        """Decode model output.
        """
        x = self.captions_dict.string(x)
        x = self.conf_decoder.decode(x)
        return self.conf_tokenizer.decode(x)

    def generate(self, model, sample):
        """Generate captions using (simple) beam search.
        """
        tgt_captions = dict()
        gen_captions = dict()

        scores, _, tokens, _ = self.generator.generate(model, sample)

        counter = 0
        for i, tb in enumerate(tokens):
            image_id = self.image_ids[i]
            image_captions = sample['target'][i]

            for t in tb:
                counter += 1
                decoded = self.decode(t)
                tgt_captions[counter] = image_captions
                gen_captions[counter] = [{
                    'image_id': image_id,
                    'caption': decoded,
                    'id': 1
                }]

        gen_captions = self.tokenizer.tokenize(gen_captions)
        return tgt_captions, gen_captions, scores

    def forward(self, model, sample, reduce=True):
        sample_indices = sample['id']
        sample_device = sample_indices.device

        tgt_captions, gen_captions, scores = self.generate(model, sample)

        _, reward = self.scorer.compute_score(tgt_captions, gen_captions)
        reward = torch.from_numpy(reward).to(device=sample_device).view(scores.shape)

        # Mean of rewards is used as baseline rather than greedy
        # decoding (see also https://arxiv.org/abs/1912.08226).
        reward_baseline = torch.mean(reward, dim=1, keepdim=True)

        loss = -scores * (reward - reward_baseline)
        loss = loss.mean()

        sample_nsentences = sample['nsentences']
        sample_ntokens = sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'ntokens': sample_ntokens,
            'nsentences': sample_nsentences,
            'sample_size': sample_nsentences,
        }
        return loss, sample_nsentences, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs),
            'ntokens': sum(log.get('ntokens', 0) for log in logging_outputs),
            'nsentences': sum(log.get('nsentences', 0) for log in logging_outputs),
            'sample_size': sum(log.get('sample_size', 0) for log in logging_outputs)
        }
