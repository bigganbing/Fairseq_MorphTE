import os
import json
from polyglot.text import Word
import copy
import enchant


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


src = 'en'
tgt = 'it'
bpeNum = 40000

src_dir = 'examples/translation/opus-100/' + src + '-' + tgt + '-' + 'bpe' + str(bpeNum) + '/'

word2mor_src, mors_src = morSeg_bpeCorpus(file_path_bpe=src_dir + 'train.' + src, lang=src)

word2mor_tgt, mors_tgt = morSeg_bpeCorpus(file_path_bpe=src_dir + 'train.' + tgt, lang=tgt)


en_dict = enchant.Dict("en")
word2mor = copy.deepcopy(word2mor_src)
for w, m in word2mor_tgt.items():
    if w not in word2mor:
        word2mor[w] = m
    else:
        if w.endswith('@@'):
            t = w[:-2]
            flag = False
            for d in en_dict.suggest(t):
                if d.startswith(t):
                    flag = True
                    break
        else:
            flag = en_dict.check(w)

        if not flag:
            word2mor[w] = m

mor_set = list(set([mor for mors in word2mor.values() for mor in mors]))
bpe_mor_res = {'mor_set': mor_set, 'word2mor': word2mor}

json.dump(bpe_mor_res, open('mors/opus100-' + src + '-' + tgt + '_bpe' + str(bpeNum) + '-jointed_morSeg_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
