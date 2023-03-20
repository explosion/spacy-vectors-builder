<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train fastText or floret vectors

This project downloads, extracts and preprocesses texts from a number of
sources and trains vectors with [floret](https://github.com/explosion/floret).

By default, the project trains floret vectors for Korean for use in `md` and
`lg` spaCy pipelines.

Prerequisites:
- linux (it may largely work on osx but this is not tested or maintained)
- a large amount of hard drive space (e.g. ~100GB total for Korean, which has
  15GB of data in OSCAR 21.09; for English, Russian, Chinese, Spanish, etc.
  you would need multiple TB with the provided defaults)
- a workstation with a good CPU, or a lot of patience

Adjust the variables `n_process_tokenize` and `vector_thread` for your CPU.

> For a Python-only cross-platform alternative, try out the simpler
> [`pipelines/floret_wiki_oscar_vectors`](https://github.com/explosion/projects/tree/v3/pipelines/floret_wiki_oscar_vectors)
> project using Wikipedia and OSCAR 2019.

## Text Sources

- Wikipedia: https://dumps.wikimedia.org
- OpenSubtitles: https://opus.nlpl.eu/OpenSubtitles-v2018.php (https://www.opensubtitles.org)
- WMT Newscrawl: https://data.statmt.org/news-crawl/
- OSCAR 21.09: https://oscar-corpus.com/post/oscar-v21-09/

OpenSubtitles and WMT Newscrawl only contain texts for a small subset of the
languages included in Wikipedia or OSCAR, so you may need to remove the
assets and adjust/remove related steps to use a subset of the resources.

### Source Requirements

#### Wikipedia

Install `Wikiparsec`: https://github.com/rspeer/wikiparsec

Choose a current version available at https://dumps.wikimedia.org for this
language or switch to `"latest"`.

#### OSCAR 21.09

The dataset [`oscar-corpus/OSCAR-2109`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2109) requires you to:
- create a Hugging Face Hub account
- agree to the dataset terms to access: https://huggingface.co/datasets/oscar-corpus/OSCAR-2109
- authenticate with `huggingface-cli login`

#### OSCAR 2019

As an alternative to OSCAR 21.09, you can stream from
[`oscar`](https://huggingface.co/datasets/oscar) without authentication.

## floret Parameters

[floret](https://github.com/explosion/floret) has a large number of
parameters and it's difficult to give advice for all configurations, but the
parameters described here are the ones that it makes sense to customize for
any new language and to experiment with initially.

Be aware that if you're using more than one thread, the results of each run
with fastText or floret will be slightly different.

### `vector_minn` / `vector_maxn`

The minimum and maximum character n-gram lengths should be adapted for the
language and writing system. The n-grams should capture common grammatical
affixes like English `-ing`, without making the number of n-grams per word
too large. Very short n-grams aren't meaningful and very long n-grams will be
too sparse and won't be useful for cases with misspellings and noise.

A good rule of thumb is that `maxn` should correspond to the length of the
longest common affix + `1`, so for many languages with alphabets, `minn
4`/`maxn 5` can be a good starting point, similar to `minn 5`/`maxn 5`, which
was shown to be a reasonable default for the [original fastText
vectors](https://fasttext.cc/docs/en/crawl-vectors.html).

For writing systems where one character corresponds to a syllable, shorter
n-grams are typically more suitable. For Korean, where each (normalized)
character is a syllable and most grammatical affixes are 1-2 characters,
`minn 2`/`maxn 3` seems to perform well.

### `vector_bucket_md` / `vector_bucket_lg`

The bucket size is the number of rows in the floret vector table. For
tagging and parsing, a bucket size of 50k performs well, but larger sizes may
still lead to small improvements. For NER, the performance continues to
improve for bucket sizes up to at least 200k.

In a spaCy pipeline package, 50k 300-dim vectors are ~60MB and 200k 300-dim
vectors are ~230MB.

### `vector_hash_count`

The recommended hash count is `2`, especially for smaller bucket sizes.

Larger hash counts are slower to train with floret and slightly slower in
inference in spaCy, but may lead to slightly improved performance, especially
with larger bucket sizes.

### `vector_epoch`

You may want to reduce the number of epochs for larger training input sizes.

### `vector_min_count`

You may want to increase the minimum word count for larger training input
sizes.

### `vector_lr`

You may need to decrease the learning rate for larger training input sizes to
avoid NaN errors, see:
https://fasttext.cc/docs/en/faqs.html#im-encountering-a-nan-why-could-this-be

### `vector_thread`

Adjust the number of threads for your CPU. With a larger number of threads,
you may need more epochs to reach the same performance.

## Notes

The project does not currently clean up any intermediate files so that it's
possible to resume from any point in the workflow. The overall disk space
could be reduced by cleaning up files after each step, keeping only the final
floret input text file. floret does require the input file to be on disk
during training.

floret always writes the full `.bin` and `.vec` files after training. These
may be 5GB+ each even though the final `.floret` table is much smaller.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `extract-wikipedia` | Convert Wikipedia XML to plain text with Wikiparsec |
| `tokenize-wikipedia` | Tokenize Wikipedia |
| `extract-opensubtitles` | Extract OpenSubtitles data |
| `tokenize-opensubtitles` | Tokenize OpenSubtitles |
| `tokenize-oscar` | Tokenize and sentencize oscar dataset |
| `create-input` | Concatenate tokenized input texts |
| `compile-floret` | Compile floret |
| `train-floret-vectors-md` | Train floret md vectors |
| `train-floret-vectors-lg` | Train floret lg vectors |
| `train-fasttext-vectors` | Train fastText vectors |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prepare-text` | `extract-wikipedia` &rarr; `tokenize-wikipedia` &rarr; `extract-opensubtitles` &rarr; `tokenize-opensubtitles` &rarr; `tokenize-oscar` &rarr; `create-input` |
| `train-vectors` | `compile-floret` &rarr; `train-floret-vectors-md` &rarr; `train-floret-vectors-lg` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `software/floret` | Git |  |
| `/scratch/vectors/downloaded/wikipedia/slwiki-20230220-pages-articles.xml.bz2` | URL |  |
| `/scratch/vectors/downloaded/opensubtitles/sl.txt.gz` | URL |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
