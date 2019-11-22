import torch

from fairseq import options, tasks, checkpoint_utils
from fairseq.data import encoders

import data


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

    image_ids = data.read_image_ids('output/valid-ids.txt')

    if args.features == 'grid':
        image_ds = data.GridFeaturesDataset('output/valid-features-grid', image_ids)
    else:  # args.features == 'obj':
        image_md = data.read_image_metadata('output/valid-features-obj/metadata.csv')
        image_ds = data.ObjectFeaturesDataset('output/valid-features-obj', image_ids, image_md)

    sample_ids = [line.rstrip('\n') for line in open(args.input)]

    for sample_id in sample_ids:
        features, locations = image_ds.read_data(int(sample_id))
        length = features.shape[0]

        if use_cuda:
            features = features.cuda()
            locations = locations.cuda()

        sample = {
            'net_input': {
                'src_tokens': features.unsqueeze(0),
                'src_locations': locations.unsqueeze(0),
                'src_lengths': [length]
            }
        }

        translations = task.inference_step(generator, models, sample)
        prediction = decode(captions_dict.string(translations[0][0]['tokens']))

        print(f'{sample_id}: {prediction}')


def cli_main():
    parser = options.get_generation_parser(interactive=True, default_task='captioning')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
