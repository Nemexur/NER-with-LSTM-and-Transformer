from typing import (
    List, Callable, Any
)
import re
from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool, cpu_count


def perform_parallel(
    data: List[Any],
    func: Callable,
    desc: str,
    n_jobs: int,
    verbose: bool = True
) -> List[Any]:
    """
    Perform function in parallel.

    Parameters
    ----------
    data : `List`, required
        List of values to process.
    func : `Callable`, required
        Function which is going to be used for processing.
        It may accept only one parameter.
    desc : `str`, required
        Description for tqdm when data is being processed.
    n_jobs : `int`, required
        Specifies the number of kernels to run build.
        If 0 or 1 parallel execution is not considerate.
        If -1 all kernels are used.
    verbose : `bool`, optional (default = `True`)
        Whether to show progress or not.

    Returns
    -------
    `List`
        List of function processed values.
    """
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            # Если передали количество меньшее,
            # чем количество n_jobs, то может быть 0
            chunksize = max(1, len(data) // n_jobs)
            logger.info(
                f'Performing multiprocessing on {n_jobs} '
                f'kernels with chunksize {chunksize}'
            )
            return list(tqdm(
                p.imap(func, data, chunksize=chunksize),
                desc=desc,
                disable=not verbose,
                total=len(data)
            ))
    else:
        return [func(x) for x in tqdm(data, desc=desc)]


def replace_numbers(in_file_at_path: str, replacement: str = '[NUM]') -> str:
    """
    Replace all numbers in file with replacement.

    Parameters
    ----------
    in_file_at_path : `str`, required
        File in which to perform replacing.

    replacement : `str`, optional (default = `'[NUM]'`)
        String on which we will replace numbers.

    Returns
    -------
    `str`
        File as txt with replaced numbers.
    """
    with open(in_file_at_path, 'r', encoding='utf-8') as file:
        txt = file.read()
        new_txt = re.sub(r'\d+', replacement, txt, flags=re.I)
        return new_txt


def create_sentencepiece_dataset(path: str, save_path: str) -> None:
    """
    Function to create dataset for SentencePiece BPE tokenizer.

    Parameters
    ----------
    path : `str`, required
        Path with dataset in ConLL2003 dataset.
    save_path : `str`, required
        Path to save dataset for
        SentencePiece BPE tokenizer.
    """
    # Reading file
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for x in file:
            if x == '\n': data.append('\n')
            else:
                line_split = x.split()
                data.append(f'{line_split[0]} ')
    txt = ''.join(data)
    # Saving dataset
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(txt)
