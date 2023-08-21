import random

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
import spacy
import sys
from codebook.utils import diffStrings

sys.path.append('../')

from codebook.utils import readCoNLL12

nlp = spacy.load('en_core_web_sm')


def punctuation(pdata):
    ret = Perturb.perturb(pdata, Perturb.punctuation, keep_original=False)
    return [x for l in ret.data for x in l]


def typos(data):
    ret = Perturb.perturb(data, Perturb.add_typos, nsamples=10, keep_original=False)
    return [x for l in ret.data for x in l]


def contraction(data):
    ret = Perturb.perturb(data, Perturb.contractions, keep_original=False)
    return [x for l in ret.data for x in l]


def changeName(pdata):
    ret = Perturb.perturb(pdata, Perturb.change_names, keep_original=False)
    return [x for l in ret.data for x in l]


def changeLocation(pdata):
    ret = Perturb.perturb(pdata, Perturb.change_location, keep_original=False)
    return [x for l in ret.data for x in l]


def negation(pdata):
    ret = Perturb.perturb(pdata, Perturb.add_negation, keep_original=False)
    return [x for l in ret.data for x in l]


def perturbe_all(data):
    pdata = list(nlp.pipe(data))
    gens = []

    gens.extend(punctuation(pdata))
    gens.extend(typos(data))

    gens.extend(contraction(data))
    gens.extend(changeName(pdata))
    gens.extend(changeLocation(pdata))
    # gens.extend(negation(pdata))

    return gens


def generate(sent: str, max_num: int = 10):
    pdata = list(nlp.pipe([sent]))
    gens = []

    gens.extend(punctuation(pdata))
    gens.extend(typos([sent]))

    gens.extend(contraction([sent]))
    gens.extend(changeName(pdata))
    gens.extend(changeLocation(pdata))
    # gens.extend(negation(pdata))

    gens = list(set(gens))
    gens = random.sample(gens, k=max_num)

    gdtokenList = list(nlp.pipe(gens))
    gtoken = [[t.text for t in w] for w in gdtokenList]

    return gens, gtoken, [diffStrings(sent, new) for new in gens]


def massive_test():
    oriSentencePairs = readCoNLL12()
    print(oriSentencePairs[:2])

    oriSentence = [x[2] for x in oriSentencePairs]
    oriSentence = oriSentence[:100]

    print(oriSentence[:2])

    genSentences = perturbe_all(oriSentence)
    genSentences.sort()
    print('\n'.join(genSentences))
    print('Total generate: ', len(genSentences))


def simple_test():
    test_sent = 'But if not, your blessing of peace will come back to you .'
    generated_sents, generated_tokens, replaced_pairs = generate(test_sent)

    print(len(generated_sents))
    print('\n'.join(generated_sents))
    for pair in replaced_pairs:
        print('{} -> {}'.format(pair[0], pair[1]))


if __name__ == '__main__':
    editor = Editor()

    simple_test()
