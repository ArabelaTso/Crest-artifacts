import os
import re
import subprocess
from spacy_conll import init_parser

# METRIC = "blanc"
METRIC_SET = {
    "muc",
    "bcub",
    "ceafm",
    "ceafe",
    "blanc"
}


def process_command_output(st):
    pattern = "\S\s(.*?)%"
    results = re.findall(pattern, st)
    results = list(map(lambda x: float(x.split(' ')[-1]), results))
    # recall, precision, F1
    return results


def text2Conll(sent):
    # Initialise English parser, already including the ConllFormatter as a pipeline component.
    # Indicate that we want to get the CoNLL headers in the string output.
    # `use_gpu` and `verbose` are specific to stanza. These keywords arguments are passed onto their Pipeline() initialisation
    nlp = init_parser("en",
                      "stanza",
                      parser_opts={"use_gpu": False, "verbose": False},
                      include_headers=True)
    # Parse a given string
    doc = nlp(sent)

    # Get the CoNLL representation of the whole document, including headers
    conll = doc._.conll_str
    print(conll)


def sentence_level_eval_score(gold='./gold.json', predict='./predict.json', METRIC='blanc',
                              all_predict_file='./temp_all_predict.conll', all_gold_file='./temp_all_gold.conll'):
    assert METRIC in METRIC_SET

    output_gold = "{}.conll".format(gold)
    output_predict = "{}.conll".format(predict)

    os.system("python ../corefconversion-master/jsonlines2conll.py -g {} -o {}".format(predict, output_predict))
    os.system("python ../corefconversion-master/jsonlines2conll.py -g {} -o {}".format(gold, output_gold))

    output = subprocess.check_output(
        "perl ../reference-coreference-scorers-master/scorer.pl {} {} {}".format(METRIC, output_predict, output_gold),
        shell=True,
        text=True)

    os.system("cat {} >> {}".format(output_predict, all_predict_file))
    os.system("cat {} >> {}".format(output_gold, all_gold_file))

    # print(output)
    output = output.split('\n')[-3]

    result = process_command_output(output)
    # print(result)
    # print("Recall = {:.2f} | Precision: {:.2f} | F1: {:.2f}".format(result[0], result[1], result[2]))

    os.remove(output_gold)
    os.remove(output_predict)
    return result


def corpus_level_eval_score(METRIC='blanc', all_predict_file='./temp_all_predict.conll',
                            all_gold_file='./temp_all_gold.conll'):
    assert METRIC in METRIC_SET

    # No file exist
    if not os.path.exists(all_predict_file) or not os.path.exists(all_gold_file):
        return 0

    output = subprocess.check_output(
        "perl ../reference-coreference-scorers-master/scorer.pl {} {} {}".format(METRIC, all_predict_file,
                                                                                 all_gold_file),
        shell=True,
        text=True)
    output = output.split('\n')[-3]

    result = process_command_output(output)
    return result

# print(sentence_level_eval_score(gold='temp_ori.json', predict='temp_new.json', METRIC='blanc'))
