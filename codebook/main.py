import time
import os
import re
import random
import copy
import sys
from collections import defaultdict
import spacy
import itertools
import crosslingual_coreference
import nltk
from stanfordcorenlp import StanfordCoreNLP

# only download once. After it has been downloaded, please comment the below sentences out.
# nltk.download('stopwords') 
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

import json
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from coref_score import sentence_level_eval_score, quickCheckCorefConsist, isTrueBug, isCorefKeep
from utils import readCoNLL12, getStandardCoref, getDepthSet, getStanfordCoref
# from baselines.PatInv import generate as PatInvGenerate  # uncomment this line to reproduce PatInv results
# from baselines.CAT import generate as CATGenerate  # uncomment this line to reproduce CAT results
# from baselines.SIT import generate as SITGenerate  # uncomment this line to reproduce SIT results
# from baselines.baseline_checklist import generate as ChecklistGenerate  # uncomment this line to reproduce CheckList results
# from baselines.baseline_textattack import generate as TextAttackGenerate # uncomment this line to reproduce TextAttack results
from baselines.crest_core import generate as CrestGenerate

sys.path.append('./')
sys.path.append('../')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seed for reproduction
random.seed(10)

stopWords = set(stopwords.words('english'))

# tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


coreNLP = StanfordCoreNLP('http://localhost', port=9001, lang="en")

checkDepth = False

CRSYS = 0 # 0: neural, 1: statistics, 2: deterministic

def setup_nlp_pipeline():
    """
    NOTE: to enable the set up, please make sure:
     - Spacy model has been installed (`python -m spacy download en_core_web_sm`)
     - nltk corpus has been installed `import nltk`
                                       `nltk.download('stopwords')`
                                       `nltk.download('averaged_perceptron_tagger')`
                                       then download
    """
    try:
        # install the model by running `python -m spacy download en_core_web_sm`
        nlp = spacy.load('en_core_web_sm')
    except:
        raise FileNotFoundError("Cannot find spacy model. Run `python -m spacy download en_core_web_sm` to download.")

    nlp.add_pipe(
        "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": -1}
    )
    print('Pipeline set up!')
    return nlp


def printCorefText(doc, coref):
    coref_text = []
    for cid, cluster in enumerate(coref, 1):
        cur_cluster = []
        for item in cluster:
            start, end = item
            try:
                cur_cluster.append(doc[start:end + 1].text)
            except:
                cur_cluster.append(' '.join(doc[start:end + 1]))
        print(f"- Cluster {cid}: {', '.join(cur_cluster)}")
        coref_text.append(cur_cluster)
    return coref_text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_syn_ant(word, tag):
    wn_pos = get_wordnet_pos(tag)
    synset = wn.synsets(word, pos=wn_pos)

    synonyms = []
    antonyms = []

    for syn in synset:
        # print(syn, syn.definition())
        for l in syn.lemmas():
            # wn.synset(l.name())
            # print("similarity: ", syn.lch_similarity())
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    synonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, synonyms))
    antonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, antonyms))

    synonyms -= {word}
    antonyms -= {word}

    # print("\nSynonyms of {}: [{}]".format(word, ','.join(synonyms)))
    # print("\nAntonyms of {}: [{}]".format(word, ','.join(antonyms)))

    return list(synonyms), list(antonyms)


def getCorefRelatedIndex(sent, coref, pos_inf):
    def getPRPCorefIndex(coref_list):
        corefIndex = set()
        for cluster in coref_list:
            # print('cluster', cluster)
            for item in cluster and item[0] == item[1]:
                if pos_inf[item[0]][1] == 'PRP':
                    corefIndex.add(item[0])
                    # corefIndex |= set([x for x in range(item[0], item[1] + 1)])
        return corefIndex

    depTree = coreNLP.dependency_parse(sent)
    corefIndex = getPRPCorefIndex(coref)

    FixedIndex = set()
    for item in depTree[1:]:
        relation, fromID, toID = item
        if toID in corefIndex:
            FixedIndex.add(fromID)

    FixedIndex |= corefIndex

    return FixedIndex


def generate_SIT(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = SITGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_CAT(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = CATGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_PatInv(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = PatInvGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_Checklist(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = ChecklistGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_Textattack(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = TextAttackGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced

def generate_Crest(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced, perps = CrestGenerate(oriTokens, detokenizer.detokenize(oriTokens), coref)
    return new_sentences, new_tokens, replaced, perps


def calMUC(predicted_clusters, gold_clusters):
    """
    the link based MUC

    Parameters
    ------
        predicted_clusters      list(list)       predicted clusters
        gold_clusters           list(list)       gold clusters
    Return
    ------
        tuple(float)    precision, recall, F1, TP,
    """

    def processCluster(coref):
        newCoref = []
        for cluster in coref:
            cur_cluster = []
            for item in cluster:
                cur_cluster.append('-'.join([str(x) for x in item]))
            newCoref.append(cur_cluster)
        return newCoref

    predicted_clusters = processCluster(predicted_clusters)
    gold_clusters = processCluster(gold_clusters)

    pred_edges = set()
    for cluster in predicted_clusters:
        pred_edges |= set(itertools.combinations(sorted(cluster), 2))
    gold_edges = set()
    for cluster in gold_clusters:
        gold_edges |= set(itertools.combinations(sorted(cluster), 2))
    correct_edges = gold_edges & pred_edges

    precision = len(correct_edges) / len(pred_edges) if len(pred_edges) != 0 else 0.0
    recall = len(correct_edges) / len(gold_edges) if len(gold_edges) != 0 else 0.0
    f1 = getF1(precision, recall)

    return precision, recall, f1


def getF1(precision, recall):
    return precision * recall * 2 / (precision + recall) if precision + recall != 0 else 0.0


def processCluster(coref):
    newCoref = []
    for cluster in coref:
        cur_cluster = []
        for item in cluster:
            cur_cluster.append('-'.join([str(x) for x in item]))
        newCoref.append(cur_cluster)
    return newCoref


def evaluate(docID, oriToken, oriCluster, newToken, newCluster, METRIC="blanc", extra_label=None):
    def cluster2json(docID, tokens, cluster):
        return {"doc_key": docID, "sentences": [tokens], "clusters": cluster}

    oriJson = cluster2json(docID, oriToken, oriCluster)
    newJson = cluster2json(docID, newToken, newCluster)

    oriFilename = "temp_ori.json" if extra_label is None else f"temp_ori_{extra_label}.json"
    newFilename = "temp_new.json" if extra_label is None else f"temp_new_{extra_label}.json"

    conll_dir = "../Output/conll"
    os.makedirs(conll_dir, exist_ok=True)
    all_predict_file = os.path.join(conll_dir, "all_predict.conll") if extra_label is None \
        else os.path.join(conll_dir, f"all_predict_{extra_label}.conll")
    all_gold_file = os.path.join(conll_dir, "all_gold.conll") if extra_label is None \
        else os.path.join(conll_dir, f"all_gold_{extra_label}.conll")

    with open(oriFilename, 'w') as f:
        json.dump(oriJson, f)
    with open(newFilename, 'w') as f:
        json.dump(newJson, f)

    result = sentence_level_eval_score(oriFilename, newFilename, METRIC=METRIC,
                                       all_predict_file=all_predict_file, all_gold_file=all_gold_file)

    os.remove(oriFilename)
    os.remove(newFilename)
    return result[1], result[0], result[2]


def Coref_testing(nlp, oriSentencePairs, genMethodName='WordReplace_Baseline', output_fn='../output.tsv', from_index=0):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    
    if not os.path.exists(output_fn):
        output_tsv = open(output_fn, 'w')
        output_tsv.write("\t".join(['doc_id', 'OID', 'GID', 'oriSent', 'genSent', 'repToken', 
                                    'perplexity',
                                    'oriConsistent',
                                    'pairConsistent',
                                    'goldCoref', 'goldCorefText',
                                    'oriCoref', 'oriCorefText',
                                    'newCoref', 'newCorefText',
                                    'isTrueBug', 
                                    'isCorefKeep', 
                                    # 'oriPrecision', 'oriRecall',
                                    # 'pairPrecision', 'pairRecall',
                                    'oriDepth', 'newDepth',
                                    'depthConsistent'
                                    ]) + '\n')
    else:
        output_tsv = open(output_fn, 'a')

    # initialize stats variables
    num_generated_pairs = 0
    num_origin_fail = 0
    num_fail = 0
    total_elapsed_time = 0.0

    OID = from_index
    oriSentencePairs = oriSentencePairs[OID:]
    # For every original sentence
    for doc_id, oriTokens, oriSent, goldCoref in oriSentencePairs:
        try:
            # analyze original sentence
            oriDoc = nlp(oriSent)

            if CRSYS == 0:
                oriCoref = getStanfordCoref(oriSent, corefAlg='neural')
            elif CRSYS == 1:
                oriCoref = getStanfordCoref(oriSent, corefAlg='statistical')

            cur_ori_precision, cur_ori_recall, cur_ori_f1 = \
                evaluate(doc_id, oriTokens, oriCoref, oriSent, goldCoref, METRIC="blanc", extra_label=genMethodName)

            oriConsistent = (cur_ori_precision == 100.0 and cur_ori_recall == 100.0)
            print("oriConsistent", oriConsistent)
            oriConsistent = quickCheckCorefConsist(oriCoref, goldCoref)

            # record depth to the closet nested NPs of the corefs
            oriDept = set()
            if checkDepth:
                oriDept = getDepthSet(oriDoc, oriSent, oriCoref)

            # Generate follow-up sentences
            genMethod = getattr(sys.modules[__name__], "generate_{}".format(genMethodName))
            
            # Start timer
            start_time = time.time()
            try:
                new_sentences, new_tokens, replaced_pairs, perps = genMethod(oriTokens, goldCoref)
            except:
                new_sentences, new_tokens, replaced_pairs = genMethod(oriTokens, goldCoref)
                perps = [-1] * len(new_sentences)
            
            # End timer
            elapsed_time = time.time() - start_time
            total_elapsed_time += elapsed_time
            
            
            print('generated new_sentences:', len(new_sentences))

            GID = 0
            for newSent, newToken, replaced_pair, perplexity in zip(new_sentences, new_tokens, replaced_pairs, perps):
                
                print(f"\n<Pair {num_generated_pairs}>")
                print('Origin  : ', oriSent)
                print('Generate: ', newSent)
                print(f"Replace: {replaced_pair[0]} -> {replaced_pair[1]}")

                print("=" * 20)
                print("> Gold sentence's Coref:")
                print(goldCoref)
                goldCoref_text = printCorefText(oriTokens, goldCoref)

                print("oriConsistent", oriConsistent)

                if oriConsistent:
                    print("[Origin Pass]")
                else:
                    print("[Origin Fail]")

                newDoc = nlp(newSent)

                # get coreference of generated sentence
                if CRSYS == 0:
                    newCoref = getStanfordCoref(newSent)
                elif CRSYS == 1:
                    newCoref = getStanfordCoref(newSent)

                # cur_new_precision, cur_new_recall, cur_new_f1 = \
                #     evaluate(doc_id, newToken, newCoref, oriTokens, oriCoref, METRIC="blanc", extra_label=genMethodName)

                # pairConsistent = (oriCoref == newCoref and len(oriCoref) > 0 and len(newCoref) > 0)
                # pairConsistent = (cur_new_precision == 100.0 and cur_new_recall == 100.0)
                pairConsistent = quickCheckCorefConsist(oriCoref, newCoref)
                
                isTB = isTrueBug(newSent, goldCoref, oriCoref, newCoref, goldCoref_text)
                
                isCK = isCorefKeep(newSent, goldCoref_text)

                # calculate depth
                newDept = set()
                if checkDepth:
                    newDept = getDepthSet(newDoc, newSent, oriCoref)

                # check depth
                depthConsistent = newDept == oriDept

                # Start printing information
                print("> Origin sentence's Coref:")
                print(oriCoref)
                oriCoref_text = printCorefText(oriDoc, oriCoref)

                print("\n> Generated sentence's Coref:")
                print(newCoref)
                newCoref_text = printCorefText(newDoc, newCoref)

                if checkDepth:
                    print('Origin Depth: \n{}'.format(str(oriDept)))
                    print('New    Depth: \n{}'.format(str(newDept)))
                    print('Depth Consist: {}'.format(str(depthConsistent)))

                # OutputVal bug report
                if not pairConsistent:
                    num_fail += 1
                    print("[Bug found!] Inconsistent!")
                else:
                    print("[Pass]")
                    
                print("=" * 20)

                num_generated_pairs += 1

                # Write test pair
                output_tsv.write('\t'.join([doc_id, str(OID), str(GID), oriSent, newSent,
                                            f'{replaced_pair[0]} -> {replaced_pair[1]}',
                                            str(perplexity),
                                            str(oriConsistent),
                                            str(pairConsistent),
                                            str(goldCoref), str(goldCoref_text),
                                            str(oriCoref), str(oriCoref_text),
                                            str(newCoref), str(newCoref_text),
                                            str(isTB), 
                                            str(isCK),
                                            # str(cur_ori_precision), str(cur_ori_recall),
                                            # str(cur_new_precision), str(cur_new_recall),
                                            str(oriDept), str(newDept),
                                            str(depthConsistent)
                                            ]) + '\n')

                GID += 1
                # End of iteration of generated sentences
        except Exception as e:
            print(f'Sentence {OID} failed. Error message:')
            print(e)
        
            
        OID += 1
        # End of iteration for the current original sentence

    ################################################
    # Output summary
    print("\n\n")
    print("=" * 20)
    print("Summary: \n" \
          "Number of origin sentence: {} | Failed: {}\n" \
          "Number of generated pairs: {} | Failed: {}".format(
        len(oriSentencePairs), num_origin_fail,
        num_generated_pairs, num_fail
    ))

    output_tsv.close()
    return num_generated_pairs, total_elapsed_time


if __name__ == '__main__':
    OutputDir = './NeuralCoref/'
    # read Output
    oriSentencePairs = readCoNLL12()
    oriSentencePairs = random.sample(oriSentencePairs, k=1000)

    OVERHEAD_FILE = os.path.join(OutputDir, 'overhead_cat.csv')
    
    if not os.path.exists(OVERHEAD_FILE):
        with open(OVERHEAD_FILE, 'w') as fw:
            fw.write(','.join(['Method', 'NumberOfFollowUp', 'TotalElapsedTimeSeconds', 'AverTime']))
            fw.write('\n')
    
    # load nlp
    nlp = setup_nlp_pipeline()

    # choose generation methods
    genMethodNameList = ['SIT', 'CAT' , 'PatInv', 'Textattack', 'Crest',  'Checklist']  # 

    for genMethodName in genMethodNameList:
        print(f"\n\n>> Approach: {genMethodName} <<\n\n")
        
        # set up output tsv file
        output_tsv_fn = os.path.join(OutputDir, '{}.tsv'.format(genMethodName))
        try:
            num_generated_pairs, total_elapsed_time = Coref_testing(nlp, oriSentencePairs, genMethodName=genMethodName, output_fn=output_tsv_fn, from_index=703)
            with open(OVERHEAD_FILE, 'a') as fw:
                fw.write(','.join([genMethodName, str(num_generated_pairs), '%.4f'%total_elapsed_time, '%.4f'%(total_elapsed_time / num_generated_pairs) if num_generated_pairs != 0 else '0.0']))
                fw.write('\n')
        except Exception as e:
            print(f"{genMethodName} failed! Error message:")
            print(e)
