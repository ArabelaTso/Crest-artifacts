import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

import os
import sys
import pickle

sys.path.append('./')
sys.path.append('../')
# os.chdir('../')
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
num_of_perturb = 10  # number of generated similar words for a given position


# Generate a list of similar sentences by Bert
def perturb(sent, bertmodel, num):
    tokens = tokenizer.tokenize(sent)
    pos_inf = nltk.tag.pos_tag(tokens)

    # the elements in the lists are tuples <index of token, pos tag of token>
    bert_masked_indexL = list()

    # collect the token index for substitution
    for idx, (word, tag) in enumerate(pos_inf):
        # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
        if (tag.startswith('NN') or tag.startswith('JJ')):
            tagFlag = tag[:2]

            # we do not perturb the first and the last token because BERT's performance drops on for those positions
            if (idx != 0 and idx != len(tokens) - 1):
                bert_masked_indexL.append((idx, tagFlag))

    bert_new_sentences, bert_new_tokens, replaced_pairs = [], [], []

    # generate similar setences using Bert
    if bert_masked_indexL:
        bert_new_sentences, bert_new_tokens, replaced_pairs = perturbBert(sent, bertmodel, num, bert_masked_indexL)

    return bert_new_sentences, bert_new_tokens, replaced_pairs


def perturbBert(sent, bertmodel, num, masked_indexL):
    new_sentences = list()
    new_tokens = []
    replaced = []

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
        # another option is to mark the unknown words as [MASK]; we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
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
                new_tokens.append(tokens)
                replaced.append((original_word, t))

        tokens[masked_index] = original_word

    return new_sentences, new_tokens, replaced


def generate(sent, max_num=num_of_perturb):
    return perturb(sent, bertmodel, max_num)


def simple_test():
    test_sent = 'But if not, your blessing of peace will come back to you .'
    generated_sents, generated_tokens, replaced_pairs = generate(test_sent)

    print(len(generated_sents))
    print('\n'.join(generated_sents))
    print('\n'.join(generated_sents))
    for pair in replaced_pairs:
        print('{} -> {}'.format(pair[0], pair[1]))



if __name__ == '__main__':
    simple_test()
    # oriSentencePairs = readCoNLL12()
    # # since we fix the random seed, the sampling result will be the same
    # oriSentencePairs = random.sample(oriSentencePairs, k=200)
    #
    # gen_dict = defaultdict(list)
    # plain_text = open('./baselines/output/SITOutput.txt', 'w')
    #
    # cnt = 0
    # for doc_id, oriTokens, oriSent, goldCoref in oriSentencePairs:
    #     generated_sents, generated_tokens, replaced_pairs = generate(oriSent)
    #     gen_dict[oriSent] = generated_sents
    #
    #     plain_text.write(oriSent + '\n')
    #     plain_text.write('\n'.join(generated_sents))
    #     if len(generated_sents):
    #         plain_text.write('\n')
    #
    #     print(f'Finish Sentence {cnt}, generated {len(generated_sents)}.')
    #     cnt += 1
    #
    # plain_text.close()
    #
    # with open('./baselines/output/SITGen.dat', 'wb') as f:
    #     pickle.dump(gen_dict, f)
