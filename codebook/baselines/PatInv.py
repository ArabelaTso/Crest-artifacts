import os
import re
import sys
import os
sys.path.append('./')
sys.path.append('../')

import nltk
import pickle
import torch
from apted.helpers import Tree
from apted import APTED
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from codebook.utils import readCoNLL12
import random
from collections import defaultdict

# fix the random seed so that the sampling result will be the same
random.seed(10)

# use nltk treebank tokenizer and detokenizer
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

# BERT initialization
berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
bertmodel.eval()

# parameters
num_of_perturb = 50  # number of generated similar words for a given position

stopWords = list(set(stopwords.words('english')))


def get_wordnet_pos(word):
    """Get pos tags of words in a sentence"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def treeToTree(tree):
    """Compute the distance between two trees by tree edit distance"""
    tree = tree.__str__()
    tree = re.sub(r'[\s]+', ' ', tree)
    tree = re.sub('\([^ ]+ ', '(', tree)
    tree = tree.replace('(', '{').replace(')', '}')
    return next(map(Tree.from_text, (tree,)))


def treeDistance(tree1, tree2):
    """Compute distance between two trees"""
    tree1, tree2 = treeToTree(tree1), treeToTree(tree2)
    ap = APTED(tree1, tree2)
    return ap.compute_edit_distance()


# Generate a list of similar sentences by Bert
def perturb(sent, bertmodel, num):
    tokens = tokenizer.tokenize(sent)
    pos_inf = nltk.tag.pos_tag(tokens)

    # the elements in the lists are tuples <index of token, pos tag of token>
    bert_masked_indexL = list()

    # collect the token index for substitution
    for idx, (word, tag) in enumerate(pos_inf):
        if (tag.startswith("JJ") or tag.startswith("JJR") or tag.startswith("JJS")
                or tag.startswith("PRP") or tag.startswith("PRP$") or tag.startswith("RB")
                or tag.startswith("RBR") or tag.startswith("RBS") or tag.startswith("VB") or
                tag.startswith("VBD") or tag.startswith("VBG") or tag.startswith("VBN") or
                tag.startswith("VBP") or tag.startswith("VBZ") or tag.startswith("NN") or
                tag.startswith("NNS") or tag.startswith("NNP") or tag.startswith("NNPS")):
            tagFlag = tag[:2]

            # we do not perturb the first and the last token because BERT's performance drops on for those positions
            if (idx != 0 and idx != len(tokens) - 1):
                bert_masked_indexL.append((idx, tagFlag))

    bert_new_sentences = list()

    # generate similar sentences using Bert
    if bert_masked_indexL:
        bert_new_sentences = perturbBert(sent, bertmodel, num, bert_masked_indexL)

    return {sent: bert_new_sentences}


def perturbBert(sent, bertmodel, num, masked_indexL):
    new_sentences = list()
    tokens = tokenizer.tokenize(sent)

    invalidChars = set(string.punctuation)

    # for each idx, use Bert to generate k (i.e., num) candidate tokens
    for (masked_index, tagFlag) in masked_indexL:
        original_word = tokens[masked_index]

        low_tokens = [x.lower() for x in tokens]
        low_tokens[masked_index] = '[MASK]'

        # try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = berttokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            prediction = bertmodel(tokens_tensor)

        # skip the sentences that contain unknown words
        # another option is to mark the unknow words as [MASK]; we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print(f'skip a sentence. unknown token is %s' % error)
            break

        # get the similar words
        topk_Idx = torch.topk(prediction[0, masked_index], num)[1].tolist()
        topk_tokens = berttokenizer.convert_ids_to_tokens(topk_Idx)

        # remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # this step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

        # generate similar sentences
        for t in topk_tokens:
            if any(char in invalidChars for char in t):
                continue
            tokens[masked_index] = t
            new_pos_inf = nltk.tag.pos_tag(tokens)

            # only use the similar sentences whose similar token's tag is still NN or JJ
            if (new_pos_inf[masked_index][1].startswith(tagFlag)):
                new_sentence = detokenizer.detokenize(tokens)
                new_sentences.append(new_sentence)

        tokens[masked_index] = original_word

    return new_sentences


def filtering_via_syntactic_and_semantic_information_replace(pert_sent, synonyms):
    """Filter sentences by synonyms and constituency structure for PaInv-Replace.
    Returns a dictionary of original sentence to list of filtered sentences
    """
    stopWords = list(set(stopwords.words('english')))

    filtered_sent = {}
    new_token = {}
    replaced = {}

    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    # Run CoreNLPPArser on local host
    eng_parser = CoreNLPParser('http://localhost:9001')

    for original_sentence in list(pert_sent.keys()):
        # Create a dictionary from original sentence to list of filtered sentences
        filtered_sent[original_sentence] = []
        new_token[original_sentence] = []
        replaced[original_sentence] = []

        tokens_or = tokenizer.tokenize(original_sentence)
        # Constituency tree of source sentence
        source_tree = [i for i, in eng_parser.raw_parse_sents([original_sentence])]
        # Get lemma of each word of source sentence
        source_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(original_sentence)]
        new_sents = pert_sent[original_sentence]
        target_trees_GT = []
        num = 50
        # Generate constituency tree of each generated sentence
        for x in range(int(len(new_sents) / num)):
            try:
                target_trees_GT[(x * num):(x * num) + num] = [i for i, in eng_parser.raw_parse_sents(
                    new_sents[(x * num):(x * num) + num])]
            except Exception as e:
                print('Exception raised 1. Skip.')
                print(e)
                break

        x = int(len(new_sents) / num)
        try:
            target_trees_GT[(x * num):] = [i for i, in eng_parser.raw_parse_sents(new_sents[(x * num):])]
        except Exception as e:
            print('Exception raised 2. Skip.')
            print(e)
            break

        if len(new_sents) != len(target_trees_GT):
            continue

        for x in range(len(new_sents)):
            s = new_sents[x]
            target_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s)]
            # If sentence is same as original sentence then filter that
            if s.lower() == original_sentence.lower():
                continue
            # If there constituency structure is not same, then filter
            if treeDistance(target_trees_GT[x], source_tree[0]) > 1:
                continue
            # If original sentence and generate sentence have same lemma, then filter
            if target_lem == source_lem:
                continue
            # Tokens of generated sentence
            tokens_tar = tokenizer.tokenize(s)
            for i in range(len(tokens_or)):
                try:
                    if tokens_or[i] != tokens_tar[i]:
                        word1 = tokens_or[i]
                        word2 = tokens_tar[i]
                        word1_stem = stemmer.stem(word1)
                        word2_stem = stemmer.stem(word2)
                        word1_base = WordNetLemmatizer().lemmatize(word1, 'v')
                        word2_base = WordNetLemmatizer().lemmatize(word2, 'v')
                        # If original word and predicted word have same stem, then filter
                        if word1_stem == word2_stem:
                            continue
                        # If they are synonyms of each other, then filter
                        # syn1 = synonyms(word1_base)
                        # syn2 = synonyms(word2_base)
                        syn1 = synonyms[word1_base]
                        syn2 = synonyms[word2_base]
                        if (word1 in syn2) or (word1_base in syn2) or (word2 in syn1) or (word2_base in syn1):
                            continue
                        if ((word1 in stopWords) or (word2 in stopWords) or (word1_stem in stopWords)
                                or (word2_stem in stopWords) or (word1_base in stopWords) or (word2_base in stopWords)):
                            continue
                        filtered_sent[original_sentence].append(s)
                        new_token[original_sentence].append(tokens_tar)
                        replaced[original_sentence].append((word1, word2))
                except KeyError as error:
                    print(f'skip a sentence. unknown token is %s' % error)
                    continue
    return filtered_sent, new_token, replaced


def generate(sent, max_num=num_of_perturb):
    # because the running directory is under code/
    # print(os.path.abspath(os.curdir))
    with open("./baselines/synonyms.dat", 'rb') as f:
        synonyms = pickle.load(f)

    # generate far more sentences
    syntactically_similar_sentences = perturb(sent, bertmodel, max_num)

    # rule out the sentences with the synonyms, the same constituency structure
    filtered_sentences, filtered_tokens, replaced_pairs = filtering_via_syntactic_and_semantic_information_replace(syntactically_similar_sentences,
                                                                                  synonyms)

    return filtered_sentences[sent], filtered_tokens[sent], replaced_pairs[sent]


def simple_test():
    test_sent = 'But if not, your blessing of peace will come back to you .'
    generated_sents, generated_tokens, replaced_pairs = generate(test_sent)

    print(len(generated_sents))
    print('\n'.join(generated_sents))
    print('\n'.join(generated_sents))
    for pair in replaced_pairs:
        print('{} -> {}'.format(pair[0], pair[1]))


if __name__ == '__main__':
    oriSentencePairs = readCoNLL12()
    # since we fix the random seed, the sampling result will be the same
    oriSentencePairs = random.sample(oriSentencePairs, k=200)

    gen_dict = defaultdict(list)
    plain_text = open('./baselines/output/PatInvOutput.txt', 'w')

    cnt = 0
    for doc_id, oriTokens, oriSent, goldCoref in oriSentencePairs:
        generated_sents, generated_tokens, replaced_pairs = generate(oriSent)
        gen_dict[oriSent] = generated_sents

        plain_text.write(oriSent + '\n')
        plain_text.write('\n'.join(generated_sents))
        if len(generated_sents):
            plain_text.write('\n')

        print(f'Finish Sentence {cnt}, generated {len(generated_sents)}.')
        cnt += 1

    plain_text.close()

    with open('./baselines/output/PatInvGen.dat', 'wb') as f:
        pickle.dump(gen_dict, f)
