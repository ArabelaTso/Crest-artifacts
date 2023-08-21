import re
import pandas as pd
from nltk.parse import CoreNLPParser
import re
from apted.helpers import Tree

eng_parser = CoreNLPParser('http://localhost:9001')


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

    source_tree = [i for i, in eng_parser.raw_parse_sents([sent])]
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


def selectTest():
    tsv_path = '../Output/Crest_before_filter.tsv'
    output_tsv_path = '../Output/Crest.tsv'

    df = pd.read_csv(tsv_path, delimiter='\t')
    # df = df.iloc[:2, :]

    oriDepth, newDepth, depthConsistent = df['oriDepth'].tolist(), df['newDepth'].tolist(), df[
        'depthConsistent'].tolist()

    select = []

    for ori, new, depthC in zip(oriDepth, newDepth, depthConsistent):
        if ori == 'set()':
            select.append(True)
            continue

        ori_set = ori.replace('{', '').replace('}', '').split(',')  # {17, 3, 4}
        new_set = new.replace('{', '').replace('}', '').split(',')  # {17, 3, 4}

        if len(ori_set) != len(new_set):
            select.append(True)
        else:
            if str(depthC).lower() == 'true':
                select.append(True)
            else:
                select.append(False)

    df['keepTest'] = select

    df.to_csv(output_tsv_path, sep='\t')


if __name__ == '__main__':
    selectTest()
