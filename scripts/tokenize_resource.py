from typing import Optional
import re
import spacy
import typer
from itertools import islice
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from datasets import load_dataset
from more_itertools import chunked


def tokenize_batch(nlp, batch):
    output = []
    texts = (re.sub(r"\s+", " ", line.strip()) for line in batch)
    for doc in nlp.pipe(texts):
        for sent in doc.sents:
            output.append(" ".join([t.text for t in sent]) + "\n")
    return output


def main(
    lang: str,
    output_file: Path,
    input_file: Optional[Path] = None,
    input_dataset: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    dataset_split: Optional[str] = None,
    dataset_streaming: bool = True,
    dataset_auth_token: bool = False,
    max_texts: int = -1,
    n_process: int = 8,
    batch_size: int = 1000,
):
    if input_file is None and input_dataset is None:
        raise ValueError("Provide either an input file or an input dataset.")

    if lang == "ko":
        nlp = spacy.blank(
            "ko", config={"nlp": {"tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"}}}
        )
    elif lang == "zh":
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "pkuseg"}}})
        nlp.tokenizer.initialize(pkuseg_model="spacy_ontonotes")
    else:
        nlp = spacy.blank(lang)

    nlp.add_pipe("sentencizer")
    nlp.max_length = 10**8

    if input_file:
        if max_texts > 0:
            texts = islice(open(input_file, encoding="utf8"), max_texts)
        else:
            texts = open(input_file, encoding="utf8")
    elif input_dataset:
        dataset = load_dataset(
            input_dataset,
            dataset_subset,
            split=dataset_split,
            streaming=dataset_streaming,
            use_auth_token=dataset_auth_token,
        )
        if max_texts > 0:
            texts = (line["text"] for line in islice(iter(dataset), max_texts))
        else:
            texts = (line["text"] for line in dataset)

    with open(output_file, "w", encoding="utf-8") as output_fileh, Pool(processes=n_process) as pool:
        result = pool.imap(partial(tokenize_batch, nlp), chunked(texts, batch_size))
        for lines in result:
            output_fileh.writelines(lines)


if __name__ == "__main__":
    typer.run(main)
