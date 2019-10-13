import os
import torch
import numpy as np

from fairseq import options, tasks, checkpoint_utils
from fairseq.data import encoders


def main(args):
    print(args)
    use_cuda = torch.cuda.is_available() and not args.cpu

    task = tasks.setup_task(args)
    captions_dict = task.target_dictionary

    models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
    model = models[0]

    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )

    if torch.cuda.is_available() and not args.cpu:
        model.cuda()

    generator = task.build_generator(args)
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def decode(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    image_ids = [line.rstrip('\n') for line in open(args.input)]

    for image_id in image_ids:
        features_file = os.path.join(args.features_dir, 'valid-features', f'{image_id}.npy')
        features = np.load(features_file)

        features_tensor = torch.as_tensor(features).unsqueeze(0)
        features_lengths = torch.tensor((1, features.shape[0]), dtype=torch.int64)

        if use_cuda:
            features_tensor = features_tensor.cuda()

        sample = {
            'net_input': {
                'src_tokens': features_tensor,
                'src_lengths': features_lengths
            }
        }

        translations = task.inference_step(generator, models, sample)
        prediction = decode(captions_dict.string(translations[0][0]['tokens']))

        print(f'{image_id}: {prediction}')


def cli_main():
    parser = options.get_generation_parser(interactive=True, default_task='captioning')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
