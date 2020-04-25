from typing import (
    List
)
import itertools
from tqdm import tqdm
from loguru import logger
import sentencepiece as spm
from argparse import ArgumentParser, Namespace

SENTENCEPIECE_MODEL = spm.SentencePieceProcessor()


def main(args: Namespace):
    SENTENCEPIECE_MODEL.Load(args.sentencepiece_model_path)
    for dataset_path in args.dataset_paths:
        logger.info(
            f"Processing dataset at path: {dataset_path}",
            feautre='f-strings'
        )
        encoded_dataset = process_dataset(
            at_path=dataset_path,
            tagging_format=args.tagging_format
        )
        save_path = f"{dataset_path}.spm"
        logger.success(
            f"Successfully processed dataset and started saving at path: {save_path}",
            feature='f-strings'
        )
        with open(save_path, 'w') as file:
            file.write(encoded_dataset)
        logger.success('Saved dataset')


def process_dataset(
    at_path: str,
    tagging_format: str
) -> str:
    """
    Process dataset by encoding tokens with SentencePieceProcessor
    and respectively by adding new tags

    Parameters
    ----------
    at_path : `str`, required
        Path to the dataset in CoNLL-2003 format
    tagging_format : `str`, required
        Type of tagging in dataset.

    Returns
    -------
    `str`
        Encoded dataset as a string in CoNLL-2003 format
    """
    processed_dataset = []
    with open(at_path, "r") as data_file:
        # Group into alternative divider / sentence chunks.
        for is_divider, lines in tqdm(itertools.groupby(data_file, is_divider_line),
                                      desc='Processing lines'):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                fields = [line.strip().split() for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                fields = [list(field) for field in zip(*fields)]
                tokens, ner_tags = fields
                # Build new sample after encoding
                sample = build_sentence_piece_tokens_and_tags(
                    tokens=tokens,
                    tags=ner_tags,
                    tagging_format=tagging_format
                )
                processed_dataset.append('\n'.join(sample))
    return '\n\n'.join(processed_dataset)


def is_divider_line(line: str) -> bool:
    """
    Check whether the line is the divider
    """
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


def build_sentence_piece_tokens_and_tags(
    tokens: List[str],
    tags: List[str],
    tagging_format: str
) -> List[str]:
    """
    Encode tokens with SentencePiece and also expand tags for NER based on encoding

    Parameters
    ----------
    tokens : `List[str]`, required
        Tokens from sentence
    tags : `List[str]`, required
        List of tags for each token in sentence
    tagging_format : `str`, required
        Type of tagging in dataset.

    Returns
    -------
    `List[str]`
        List of tokens with its corresponding tags separated by space
    """
    encoded_tokens = []
    tags_for_encoded_tokens = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        encoded_token = SENTENCEPIECE_MODEL.EncodeAsPieces(str(token))
        encoded_tokens.extend([e_token for e_token in encoded_token])
        number_of_tags_to_append = len(encoded_token)
        if tagging_format == 'BIO':
            if tag.startswith('B-'):
                number_of_tags_to_append -= 1
                tags_for_encoded_tokens.append(tag)
            tags_for_encoded_tokens.extend([
                f'I-{tag[2:]}' if tag.startswith('B-') or tag.startswith('I-')
                else 'O' for _ in range(number_of_tags_to_append)
            ])
        elif tagging_format == 'IOB':
            if (
                (tag.startswith('I-') and tags[i - 1][2:] != tag[2:])
                or tag.startswith('B-')
            ):
                number_of_tags_to_append -= 1
                tags_for_encoded_tokens.append(
                    f'B-{tag[2:]}' if number_of_tags_to_append > 1 else f'I-{tag[2:]}'
                )
            tags_for_encoded_tokens.extend([
                f'I-{tag[2:]}' if tag.startswith('B-') or tag.startswith('I-')
                else 'O' for _ in range(number_of_tags_to_append)
            ])
        else:
            raise ValueError(
                'Invalid tagging_format. '
                'Only BIO and IOB supported.'
            )
    assert len(encoded_tokens) == len(tags_for_encoded_tokens), \
        "Number of tokens is not equal to number of tags"
    return [f"{word} {tag}" for word, tag in zip(encoded_tokens, tags_for_encoded_tokens)]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_paths', nargs="+", required=True,
                        help="Paths to dataset for encoding. Ordinary these are train, test and valid datasets")
    parser.add_argument("--sentencepiece_model_path", help="Path to trained SentencePiece model",
                        required=True, type=str)
    parser.add_argument("--tagging_format", required=True, type=str,
                        help="Type of tagging that is used in the dataset. Only BIO and IOB are supported")
    main(args=parser.parse_args())
