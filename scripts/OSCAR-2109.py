# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""OSCAR The Open Super-large Crawled Aggregated coRpus."""


import collections
import gzip
import json
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
The Open Super-large Crawled Aggregated coRpus is a huge multilingual corpus \
obtained by language classification and filtering of the Common Crawl corpus \
using the goclassy architecture.\
"""

_URL = "https://oscar-corpus.com"

_LICENSE = """
    These data are released under this licensing scheme
    We do not own any of the text from which these data has been extracted.
    We license the actual packaging of these data under the Creative Commons CC0 license \
    (\"no rights reserved\") http://creativecommons.org/publicdomain/zero/1.0/
    To the extent possible under law, Inria has waived all copyright \
    and related or neighboring rights to OSCAR
    This work is published from: France.

    Should you consider that our data contains material that is owned by you \
    and should therefore not be reproduced here, please:
    * Clearly identify yourself, with detailed contact data such as an address, \
    telephone number or email address at which you can be contacted.
    * Clearly identify the copyrighted work claimed to be infringed.
    * Clearly identify the material that is claimed to be infringing and \
    information reasonably sufficient to allow us to locate the material.

    We will comply to legitimate requests by removing the affected sources \
    from the next release of the corpus. \
"""

_CITATION = """\
@inproceedings{AbadjiOrtizSuarezRomaryetal.2021,
  author    = {Julien Abadji and Pedro Javier Ortiz Su{\'a}rez and Laurent Romary and Beno{\^i}t Sagot},
  title     = {Ungoliant: An optimized pipeline for the generation of a very large-scale multilingual web corpus},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-9) 2021. Limerick, 12 July 2021 (Online-Event)},
  editor    = {Harald L{\"u}ngen and Marc Kupietz and Piotr Bański and Adrien Barbaresi and Simon Clematide and Ines Pisetta},
  publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-10468},
  url       = {https://nbn-resolving.org/urn:nbn:de:bsz:mh39-104688},
  pages     = {1 -- 9},
  year      = {2021},
  abstract  = {Since the introduction of large language models in Natural Language Processing, large raw corpora have played a crucial role in Computational Linguistics. However, most of these large raw corpora are either available only for English or not available to the general public due to copyright issues. Nevertheless, there are some examples of freely available multilingual corpora for training Deep Learning NLP models, such as the OSCAR and Paracrawl corpora. However, they have quality issues, especially for low-resource languages. Moreover, recreating or updating these corpora is very complex. In this work, we try to reproduce and improve the goclassy pipeline used to create the OSCAR corpus. We propose a new pipeline that is faster, modular, parameterizable, and well documented. We use it to create a corpus similar to OSCAR but larger and based on recent data. Also, unlike OSCAR, the metadata information is at the document level. We release our pipeline under an open source license and publish the corpus under a research-only license.},
  language  = {en}
}

@article{caswell-etal-2021-quality,
       author = {{Caswell}, Isaac and {Kreutzer}, Julia and {Wang}, Lisa and {Wahab}, Ahsan and {van Esch}, Daan and {Ulzii-Orshikh}, Nasanbayar and {Tapo}, Allahsera and {Subramani}, Nishant and {Sokolov}, Artem and {Sikasote}, Claytone and {Setyawan}, Monang and {Sarin}, Supheakmungkol and {Samb}, Sokhar and {Sagot}, Beno{\^\i}t and {Rivera}, Clara and {Rios}, Annette and {Papadimitriou}, Isabel and {Osei}, Salomey and {Ortiz Su{\'a}rez}, Pedro Javier and {Orife}, Iroro and {Ogueji}, Kelechi and {Niyongabo}, Rubungo Andre and {Nguyen}, Toan Q. and {M{\"u}ller}, Mathias and {M{\"u}ller}, Andr{\'e} and {Hassan Muhammad}, Shamsuddeen and {Muhammad}, Nanda and {Mnyakeni}, Ayanda and {Mirzakhalov}, Jamshidbek and {Matangira}, Tapiwanashe and {Leong}, Colin and {Lawson}, Nze and {Kudugunta}, Sneha and {Jernite}, Yacine and {Jenny}, Mathias and {Firat}, Orhan and {Dossou}, Bonaventure F.~P. and {Dlamini}, Sakhile and {de Silva}, Nisansa and {{\c{C}}abuk Ball{\i}}, Sakine and {Biderman}, Stella and {Battisti}, Alessia and {Baruwa}, Ahmed and {Bapna}, Ankur and {Baljekar}, Pallavi and {Abebe Azime}, Israel and {Awokoya}, Ayodele and {Ataman}, Duygu and {Ahia}, Orevaoghene and {Ahia}, Oghenefego and {Agrawal}, Sweta and {Adeyemi}, Mofetoluwa},
        title = "{Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language, Computer Science - Artificial Intelligence},
         year = 2021,
        month = mar,
          eid = {arXiv:2103.12028},
        pages = {arXiv:2103.12028},
archivePrefix = {arXiv},
       eprint = {2103.12028},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210312028C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@inproceedings{ortiz-suarez-etal-2020-monolingual,
    title = "A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages",
    author = "Ortiz Su{\'a}rez, Pedro Javier  and
      Romary, Laurent  and
      Sagot, Benoit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.156",
    pages = "1703--1714",
    abstract = "We use the multilingual OSCAR corpus, extracted from Common Crawl via language classification, filtering and cleaning, to train monolingual contextualized word embeddings (ELMo) for five mid-resource languages. We then compare the performance of OSCAR-based and Wikipedia-based ELMo embeddings for these languages on the part-of-speech tagging and parsing tasks. We show that, despite the noise in the Common-Crawl-based OSCAR data, embeddings trained on OSCAR perform much better than monolingual embeddings trained on Wikipedia. They actually equal or improve the current state of the art in tagging and parsing for all five languages. In particular, they also improve over multilingual Wikipedia-based contextual embeddings (multilingual BERT), which almost always constitutes the previous state of the art, thereby showing that the benefit of a larger, more diverse corpus surpasses the cross-lingual benefit of multilingual embedding architectures.",
}

@inproceedings{OrtizSuarezSagotRomary2019,
  author    = {Pedro Javier {Ortiz Su{\'a}rez} and Benoit Sagot and Laurent Romary},
  title     = {Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019},
  editor    = {Piotr Bański and Adrien Barbaresi and Hanno Biber and Evelyn Breiteneder and Simon Clematide and Marc Kupietz and Harald L{\"u}ngen and Caroline Iliadi},
  publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-9021},
  url       = {http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215},
  pages     = {9 -- 16},
  year      = {2019},
  abstract  = {Common Crawl is a considerably large, heterogeneous multilingual corpus comprised of crawled documents from the internet, surpassing 20TB of data and distributed as a set of more than 50 thousand plain text files where each contains many documents written in a wide variety of languages. Even though each document has a metadata block associated to it, this data lacks any information about the language in which each document is written, making it extremely difficult to use Common Crawl for monolingual applications. We propose a general, highly parallel, multithreaded pipeline to clean and classify Common Crawl by language; we specifically design it so that it runs efficiently on medium to low resource infrastructures where I/O speeds are the main constraint. We develop the pipeline so that it can be easily reapplied to any kind of heterogeneous corpus and so that it can be parameterised to a wide range of infrastructures. We also distribute a 6.3TB version of Common Crawl, filtered, classified by language, shuffled at line level in order to avoid copyright issues, and ready to be used for NLP applications.},
  language  = {en}
}
"""

_BASE_DATA_PAT_FORMAT_STR = (
    "{data_dir}/{language}/"
)
_BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"


def _languages():
    """Create the sorted dictionary of language codes, and language names.

    Returns:
      The sorted dictionary as an instance of `collections.OrderedDict`.
    """
    langs = {
        "af": "Afrikaans",
        "als": "Alemanic",
        "am": "Amharic",
        "an": "Aragonese",
        "ar": "Arabic",
        "arz": "Egyptian Arabic",
        "ast": "Asturian",
        "as": "Assamese",
        "av": "Avaric",
        "azb": "South Azerbaijani",
        "az": "Azerbaijani",
        "bar": "Bavarian",
        "ba": "Bashkir",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bh": "Bihari",
        "bn": "Bengali",
        "bo": "Tibetan",
        "bpy": "Bishnupriya",
        "br": "Breton",
        "bs": "Bosnian",
        "bxr": "Russia Buriat",
        "ca": "Catalan",
        "cbk": "Chavacano",
        "ceb": "Cebuano",
        "ce": "Chechen",
        "ckb": "Central Kurdish",
        "cs": "Czech",
        "cv": "Chuvash",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "diq": "Dimli",
        "dsb": "Lower Sorbian",
        "dv": "Dhivehi",
        "el": "Modern Greek",
        "eml": "Emilian-Romagnol",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "frr": "Northern Frisian",
        "fr": "French",
        "fy": "Western Frisian",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "gl": "Galician",
        "gn": "Guarani",
        "gom": "Goan Konkani",
        "gu": "Gujarati",
        "gv": "Manx",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hsb": "Upper Sorbian",
        "ht": "Haitian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ilo": "Iloko",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jbo": "Lojban",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Central Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "krc": "Karachay-Balkar",
        "ku": "Kurdish",
        "kv": "Komi",
        "kw": "Cornish",
        "ky": "Kirghiz",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lez": "Lezghian",
        "li": "Limburgan",
        "lmo": "Lombard",
        "lo": "Lao",
        "lrc": "Northern Luri",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mai": "Maithili",
        "mg": "Malagasy",
        "mhr": "Eastern Mari",
        "min": "Minangkabau",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mrj": "Western Mari",
        "mr": "Marathi",
        "ms": "Malay",
        "mt": "Maltese",
        "mwl": "Mirandese",
        "my": "Burmese",
        "myv": "Erzya",
        "mzn": "Mazanderani",
        "nah": "Nahuatl languages",
        "nap": "Neapolitan",
        "nds": "Low German",
        "ne": "Nepali",
        "new": "Newari",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan",
        "or": "Oriya",
        "os": "Ossetian",
        "pam": "Pampanga",
        "pa": "Panjabi",
        "pl": "Polish",
        "pms": "Piemontese",
        "pnb": "Western Panjabi",
        "ps": "Pushto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "ro": "Romanian",
        "ru": "Russian",
        "rue": "Rusyn",
        "sah": "Yakut",
        "sa": "Sanskrit",
        "scn": "Sicilian",
        "sco": "Scots",
        "sd": "Sindhi",
        "sh": "Serbo-Croatian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "tyv": "Tuvinian",
        "ug": "Uighur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vec": "Venetian",
        "vi": "Vietnamese",
        "vls": "Vlaams",
        "vo": "Volapük",
        "war": "Waray",
        "wa": "Walloon",
        "wuu": "Wu Chinese",
        "xal": "Kalmyk",
        "xmf": "Mingrelian",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "zh": "Chinese",
    }
    return collections.OrderedDict(sorted(langs.items()))


class Oscar2109Config(datasets.BuilderConfig):
    """OSCAR corpus."""

    def __init__(self, language: str, deduplicated=True, **kwargs):
        """BuilderConfig for OSCAR.

        Args:
            language (str): It has to contain 2-letter or 3-letter coded strings. For example: "se", "hu", "eml"
            shuffled (bool): shuffled dataset or not. Currently only shuffled=False is supported.
            deduplicated (bool): deduplicated dataset or not.
            **kwargs: Keyword arguments forwarded to super.
        """
        # Validate the language.
        if language not in _languages():
            raise ValueError("Invalid language: %s " % language)

        # Initialize the base class.
        name = f"deduplicated_{language}" if deduplicated else f"original_{language}"
        description = f"{'Deduplicated' if deduplicated else 'Original'}, {_languages()[language]} OSCAR dataset from September 2021"
        super(Oscar2109Config, self).__init__(name=name, description=description, **kwargs)

        # Additional attributes
        self.language = language
        self.deduplicated = deduplicated
        self.base_data_path = _BASE_DATA_PAT_FORMAT_STR.format(
            language=language, data_dir="packaged" if deduplicated else "packaged_nondedup"
        )


class Oscar2109(datasets.GeneratorBasedBuilder):
    """OSCAR The Open Super-large Crawled Aggregated coRpus."""

    BUILDER_CONFIGS = [
        Oscar2109Config(  # pylint: disable=g-complex-comprehension
            language=language,
            deduplicated=True,
            version=datasets.Version("2021.9.0"),
        )
        for language in _languages()
    ] + [
        Oscar2109Config(  # pylint: disable=g-complex-comprehension
            language=language,
            deduplicated=False,
            version=datasets.Version("2021.9.0"),
        )
        for language in _languages()
    ]
    BUILDER_CONFIG_CLASS = Oscar2109Config

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("int64"),
                "text": datasets.Value("string"),
                "meta": {
                    "headers": {
                        "warc-record-id": datasets.Value("string"),
                        "warc-date": datasets.Value("string"),
                        "content-type": datasets.Value("string"),
                        "content-length": datasets.Value("int32"),
                        "warc-type": datasets.Value("string"),
                        "warc-identified-content-language": datasets.Value("string"),
                        "warc-refers-to": datasets.Value("string"),
                        "warc-target-uri": datasets.Value("string"),
                        "warc-block-digest": datasets.Value("string")
                    },
                    "offset": datasets.Value("int32"),
                    "nb_sentences": datasets.Value("int32")
                }
            }),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        checksum_path = self.config.base_data_path + _BASE_CHECKSUM_FILE_NAME.format(language=self.config.language)
        checksum_file = dl_manager.download(checksum_path)
        with open(checksum_file, encoding="utf-8") as f:
            data_filenames = [line.split()[1] for line in f if line]
            data_urls = [self.config.base_data_path + data_filename for data_filename in data_filenames]
        # sort filenames so corresponding parts are aligned
        text_files = sorted(dl_manager.download([url for url in data_urls if url.endswith(".txt.gz")]))
        metadata_files = sorted(dl_manager.download([url for url in data_urls if url.endswith(".jsonl.gz")]))
        assert len(text_files) == len(metadata_files)
        metadata_and_text_files = list(zip(metadata_files, text_files))
        for meta_path, text_path in metadata_and_text_files:
            # check that meta/text part numbers are the same
            if "part" in os.path.basename(text_path):
                assert (
                    os.path.basename(text_path).replace(".txt.gz", "").split("_")[-1]
                    == os.path.basename(meta_path).replace(".jsonl.gz", "").split("_")[-1]
                )
            else:
                assert len(metadata_and_text_files) == 1
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"metadata_and_text_files": metadata_and_text_files}),
        ]

    def _generate_examples(self, metadata_and_text_files):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for meta_path, text_path in metadata_and_text_files:
            # line offsets are per text file
            offset = 0
            logger.info("generating examples from = %s", text_path)
            # some texts contain non-Unix newlines that should not be
            # interpreted as line breaks for the line counts in the metadata
            # with readline()
            with gzip.open(open(text_path, "rb"), "rt", encoding="utf-8", newline="\n") as text_f:
                with gzip.open(open(meta_path, "rb"), "rt", encoding="utf-8") as meta_f:
                    for line in meta_f:
                        # read meta
                        meta = json.loads(line)
                        meta["headers"]["warc-identified-content-language"] = meta["headers"].get("warc-identified-content-language")
                        # go to next offset
                        while offset < meta["offset"]:
                            offset += 1
                            text_f.readline()
                        # read text
                        text_lines = [text_f.readline() for _ in range(meta["nb_sentences"])]
                        # all lines contain text (no blank lines or EOF)
                        assert all(text_lines)
                        assert "\n" not in text_lines
                        offset += meta["nb_sentences"]
                        # only strip the trailing newline
                        text = "".join(text_lines).rstrip("\n")
                        yield id_, {"id": id_, "text": text, "meta": meta}
                        id_ += 1
