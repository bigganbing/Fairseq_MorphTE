import os
import json
from tqdm import tqdm
from collections import Counter
from polyglot.text import Word


def mor_func(word, lang='en'):
    mors = Word(word, language=lang).morphemes
    mors = list(mors)
    return mors


def morSeg_bpeCorpus(file_path_bpe: str, lang='de', WithAt=False, long=True):
    print("Processing morSeg ..... ", file_path_bpe)

    word2mor = {}

    with open(file_path_bpe, encoding='utf-8') as f:
        bpe2num = {}
        for bpe in f.read().strip().split():
            bpe2num[bpe] = bpe2num.get(bpe, 0) + 1
        bpe2num = sorted(bpe2num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    num_long = 0

    for bpe, num in bpe2num:

        if bpe.endswith('@@'):
            mors = mor_func(bpe[:-2], lang=lang) + ['@@']
        else:
            mors = mor_func(bpe, lang=lang)

        if long:
            if len(mors) > 3:
                mors = [mors[0], mors[1], "".join(mors[2:])]
                num_long += 1

        if WithAt:
            for i in range(len(mors) - 1):
                mors[i] = mors[i] + '@@'

        word2mor[bpe] = mors

    mors_set = list(set([mor for mors in word2mor.values() for mor in mors]))
    print(sorted(Counter([len(bpe_li) for w, bpe_li in word2mor.items()]).items()))
    print()

    return word2mor, mors_set

src = 'de'
tgt = 'en'
src_train_file = 'examples/translation/iwslt14.bpe-10000.de-en/train.de'
tgt_train_file = 'examples/translation/iwslt14.bpe-10000.de-en/train.en'

word2mor_src, mor_set_src = morSeg_bpeCorpus(src_train_file, lang=src)
word2mor_tgt, mor_set_tgt = morSeg_bpeCorpus(tgt_train_file, lang=tgt)

result = {'mor_set_src':mor_set_src, 'word2mor_src':word2mor_src, 'mor_set_tgt':mor_set_tgt, 'word2mor_tgt':word2mor_tgt}

json.dump(result, open('mors/iwslt14deen_bpe10000_morSeg_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
