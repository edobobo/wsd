import argparse
import os
from typing import List

from tqdm import tqdm

from src.utils.dataset_to_babelnet import to_bn_id
from src.utils.wsd import RaganatoBuilder, read_from_raganato, expand_raganato_path


def main():

    args = parse_args()
    output_folder = args.output_folder
    sources = args.s
    datasets_paths = args.f

    assert not os.path.exists(output_folder)
    os.mkdir(output_folder)

    rb = RaganatoBuilder()

    for source, dp in zip(sources, datasets_paths):

        rb.open_text_section(source, source)

        for sentence_idx, (_, _, sentence) in tqdm(enumerate(read_from_raganato(*expand_raganato_path(dp))), desc=f'Reading dataset {dp}'):
            rb.open_sentence_section(str(sentence_idx))
            for j, wsdi in enumerate(sentence):
                if wsdi.labels is not None:
                    rb.add_annotated_token(wsdi.annotated_token.text, wsdi.annotated_token.lemma, wsdi.annotated_token.pos, instance_id=str(j), labels=wsdi.labels, update_id=True)
                else:
                    rb.add_annotated_token(wsdi.annotated_token.text, wsdi.annotated_token.lemma, wsdi.annotated_token.pos)

    rb.store(*expand_raganato_path(f'{output_folder}/dataset'))

    def count_instances(dp):
        n = 0
        for _, _, sentence in read_from_raganato(*expand_raganato_path(dp)):
            for wsdi in sentence:
                if wsdi.labels is not None:
                    n += 1
        return n

    found = count_instances(f'{output_folder}/dataset')
    expected = sum(map(lambda dp: count_instances(dp), datasets_paths))
    assert found == expected, f'Expected {expected} instances, but found {found} instances'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', type=str, help='Output folder where the raganato dataset will be saved')
    parser.add_argument('-s', action='append', help='Sources names (must be aligned with raganato paths)')
    parser.add_argument('-f', action='append', help='Raganato paths to datasets to merge')
    parser.add_argument('--shuffle', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()

