import os
import re
import json
import difflib
from apted.helpers import Tree
from stanfordnlp.server import CoreNLPClient
from nltk.parse import CoreNLPParser

# Set up different clients for different purposes sharing the same Stanford server
# MODEL_DIR = r'/Users/bella/Downloads/stanford-corenlp-4.4.0'
# os.environ["CORENLP_HOME"] = MODEL_DIR

nltk_parser = CoreNLPParser('http://localhost:9001')
stanford_client = CoreNLPClient(properties={'annotators': 'coref', 'coref.algorithm': 'neural'},
                       # 'statistical', 'deterministic'
                       timeout=6000, memory='16G',
                       endpoint='http://localhost:9001')


def readCoNLL12(train=False, keepDict=False):
    def clean(sent: str):
        sent = sent.strip('\n')
        sent = sent.replace(" '", "'")
        sent = sent.replace(" /.", ".")
        sent = sent.replace(" ,", ',')
        sent = sent.replace(' .', '.')
        sent = sent.replace(" - ", '-')
        sent = sent.replace("`` ", "\"")
        sent = sent.replace("''", "\"")
        sent = sent.replace("`", "\'")
        sent = sent.replace(" n't", "n't")
        sent = sent.replace(" ?", '?')
        sent = sent.replace(" ;", ',')
        sent = sent.replace(" & ", '&')
        sent = sent.replace("\"", "'")
        return sent

    def isValid(sent: str):
        # Here we remove the sentence with symbols to avoid the inconsistency caused by tokenization
        invalid_symb_set = {"-", "#", '%', '&', '*', "'", ':', "\""}
        for sym in invalid_symb_set:
            if sym in sent:
                return False
        return True

    sentencePairs = []
    num_lines = 0
    num_invalid = 0

    # news_datasets = {'bn', 'mz', 'nw'}
    news_datasets = {'wb', 'bn', 'mz', 'nw', 'bc', 'nw', 'pt', 'tc'}

    # | bn | Broadcast News (广播新闻) |
    # | mz | Magazine (Newswire, 新闻专线) |
    # | nw | Newswire (新闻专线) |

    if train:
        fn = '/ssddata/jcaoap/data/semmt_test-main/data/conll12/train.english.v4_gold_conll.sen.json'
        # fn = './data/semmt_test-main/data/conll12/train.english.v4_gold_conll.sen.json'
    else:
        fn = '/ssddata/jcaoap/data/semmt_test-main/data/conll12/test.english.v4_gold_conll.sen.json'
        # fn = './data/semmt_test-main/data/conll12/test.english.v4_gold_conll.sen.json'
    with open(fn, 'r') as conll12:
        for line in conll12.readlines():
            num_lines += 1
            sent_dict = json.loads(line)
            docID = sent_dict['doc_id']
            token = sent_dict['sentence']
            sent = " ".join(token)
            if not isValid(sent) or docID[:2] not in news_datasets or not sent.endswith('.'):
                num_invalid += 1
                continue

            sent = clean(sent)
            gold_cluster = sent_dict['cluster']

            # if keep json dict or not
            if keepDict:
                sentencePairs.append((docID, token, sent, gold_cluster, sent_dict))
            else:
                sentencePairs.append((docID, token, sent, gold_cluster))
    print(
        f"Read {num_lines} lines from conll12/test.english.v4_gold_conll.sen.json."
        f"\nFiltered {num_invalid} invalid lines, "
        f"remaining {len(sentencePairs)}.")

    return sentencePairs


def readCoNLL12_val():
    def clean(sent: str):
        sent = sent.strip('\n')
        sent = sent.replace(" '", "'")
        sent = sent.replace(" /.", ".")
        sent = sent.replace(" ,", ',')
        sent = sent.replace(' .', '.')
        sent = sent.replace(" - ", '-')
        sent = sent.replace("`` ", "\"")
        sent = sent.replace("''", "\"")
        sent = sent.replace("`", "\'")
        sent = sent.replace(" n't", "n't")
        sent = sent.replace(" ?", '?')
        sent = sent.replace(" ;", ',')
        sent = sent.replace(" & ", '&')
        sent = sent.replace("\"", "'")
        return sent

    def isValid(sent: str):
        # TODO: warning! Here we remove the sentence with symbols to avoid the inconsistency caused by tokenization
        invalid_symb_set = {"-", "#", '%', '&', '*', '\'', ':', '\"'}
        for sym in invalid_symb_set:
            if sym in sent:
                return False
        return True

    sentencePairs = []
    num_lines = 0
    num_invalid = 0
    with open('../data/semmt_test-main/data/conll12/dev.english.v4_gold_conll.sen.json', 'r') as conll12:
        for line in conll12.readlines():
            num_lines += 1
            sent_dict = json.loads(line)
            docID = sent_dict['doc_id']
            token = sent_dict['sentence']
            sent = " ".join(token)
            if not isValid(sent):
                num_invalid += 1
                continue
            sent = clean(sent)
            gold_cluster = sent_dict['cluster']

            sentencePairs.append((docID, token, sent, gold_cluster))
    print(
        f"Read {num_lines} lines from conll12/test.english.v4_gold_conll.sen.json."
        f"\nFiltered {num_invalid} invalid lines, "
        f"remaining {len(sentencePairs)}.")

    return sentencePairs


def diffStrings(s1, s2):
    output_list = [li for li in difflib.ndiff(s1, s2) if li[0] != ' ']

    oldL = [x[-1] for x in output_list if x.startswith('-')]
    old = ''.join(oldL)

    newL = [x[-1] for x in output_list if x.startswith('+')]
    new = ''.join(newL)

    return (old, new)


# client.start()

def treeToTree(tree):
    """Compute the distance between two trees by tree edit distance"""
    tree = tree.__str__()
    tree = re.sub(r'[\s]+', ' ', tree)
    tree = re.sub('\([^ ]+ ', '(', tree)
    tree = tree.replace('(', '{').replace(')', '}')
    return next(map(Tree.from_text, (tree,)))


def getDepth(sent: str, token: list):
    # Only calculate the distance from the root to the closet nested NP
    # For example,
    # (ROOT
    #   (S
    #     (NP (DT The) (NN fish))
    #     (VP
    #       (VBD ate)
    #       (NP (DT the) (NN worm))
    #       (SBAR
    #         (IN because)
    #         (S (NP (PRP it)) (VP (VBZ is) (ADJP (JJ tasty))))))
    #     (. .)))
    # The depth of the noun phrase "The fish" is 3.

    source_tree = [i for i, in nltk_parser.raw_parse_sents([sent])]
    treeS = treeToTree(source_tree[0]).__str__()

    depthS = set()
    pattern = r'{}'.format('.*?'.join(token))
    targetIds = [m.start(0) for m in re.finditer(pattern, treeS)]

    # print('targetIds', targetIds)
    for ind in targetIds:
        # ind = treeS.find(tar)
        # print('index: ', ind)
        depth = len(re.findall(r'[{]', treeS[:ind])) - len(re.findall(r'[}]', treeS[:ind]))
        # print('depth', depth)
        depthS.add(depth - 1)

    return depthS


def getDepthSet(oriDoc, oriSent, oriCoref):
    oriDept = set()
    for cluster in oriCoref:
        for ent in cluster:
            # print([x.text for x in oriDoc[ent[0]: ent[1]+1]])
            oriDept |= getDepth(oriSent, [x.text for x in oriDoc[ent[0]: ent[1] + 1]])
    return oriDept


def getStandardCoref(doc):
    # assert pipeline is not None
    # doc = pipeline(text)

    return doc._.coref_clusters


def getStanfordCoref(sent, corefAlg='neural'):
    assert corefAlg in ('neural', 'statistical', 'deterministic')
    
    doc = stanford_client.annotate(sent)

    coref = []

    for x in doc.corefChain:
        cluster = []
        # print('x = ', x)
        for y in x.mention:
            # the end index - 1 is because the ground truth include the end index
            # For example, "John, a boy who ...".
            # The index of "John" is [0, 0] in the ground truth
            # While the output of StanfordCorNLP exclude the end index
            # So StanfordCorNLP outputs [0, 1] for "John".
            # So we let it -1 to be consistent with the ground truths
            cluster.append([y.beginIndex, y.endIndex - 1])
            # print('y = ', y)
            # print('y = ', y.beginIndex, y.endIndex)
        coref.append(cluster)
    return coref


def getDeterministicCoref(sent):
    doc = stanford_client.annotate(sent)

    coref = []

    for x in doc.corefChain:
        cluster = []
        # print('x = ', x)
        for y in x.mention:
            cluster.append([y.beginIndex, y.endIndex - 1])
            # print('y = ', y)
            # print('y = ', y.beginIndex, y.endIndex)
        coref.append(cluster)
    return coref


def filter_Crest(nlp, new_sentences, new_tokens, replaced_pairs, oriDept):
    filtered_sentences, filtered_tokens, filtered_pairs, new_corefs = [], [], [], []
    for newSent, newToken, replaced_pair in zip(new_sentences, new_tokens, replaced_pairs):
        newDoc = nlp(newSent)

        # get coreference of generated sentence
        newCoref = getStandardCoref(newDoc)

        newDept = getDepthSet(newDoc, newSent, newCoref)

        # check depth
        depthConsistent = newDept == oriDept

        if depthConsistent:
            filtered_sentences.append(newSent)
            filtered_tokens.append(newToken)
            filtered_pairs.append(replaced_pair)
            new_corefs.append(newCoref)

    return filtered_sentences, filtered_tokens, filtered_pairs, new_corefs
