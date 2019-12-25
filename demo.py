import os
import data
import torch

from fairseq import options, tasks, checkpoint_utils
from fairseq.data import encoders

from PIL import Image


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

    transform = data.default_transform()

    image_ids_path_dict = data.read_split_image_ids_and_paths_dict('valid')

    with open(args.input) as f:
        sample_ids = [line.rstrip('\n') for line in f]

    for sample_id in sample_ids:
        sample_path = os.path.join(args.ms_coco_dir, 'images', image_ids_path_dict[int(sample_id)])
        with Image.open(sample_path).convert('RGB') as img:
            img_tensor = transform(img).unsqueeze(0)

        src_tokens = torch.zeros(1, 0)

        if use_cuda:
            img_tensor = img_tensor.cuda()
            src_tokens = src_tokens.cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'source': img_tensor,
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
