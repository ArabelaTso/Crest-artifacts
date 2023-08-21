import re
import numpy as np
import string
import torch
import torch.nn.functional as F
from copy import deepcopy
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import nltk
from nltk.corpus import wordnet as wn

from stanfordcorenlp import StanfordCoreNLP
stanfordParser = StanfordCoreNLP('http://localhost', port=9001, lang="en")

import os
import sys
import pickle

sys.path.append('./')
sys.path.append('./baselines')
from codebook.utils import readCoNLL12
from codebook.personal_pronouns import pps
import random

# fix the random seed so that the sampling result will be the same
random.seed(10)


K_Number = 100
Max_Mutants = 50
lcache = []

def getCorefRelatedIndex(sent, coref, pos_inf):
    # print("pos_inf", pos_inf)
    # print("coref", coref)
    def getPRPCorefIndex(coref_list):
        corefIndex = set()
        PRIndex = set()
        for cluster in coref_list:
            for item in cluster:
                if item[0] == item[1] and pos_inf[item[0]][1] == 'PRP':  # consider the pronouns
                    PRIndex.add(item[0])
                corefIndex |= set([x for x in range(item[0], item[1] + 1)])
        return PRIndex

    depTree = stanfordParser.dependency_parse(sent)
    tokens = stanfordParser.word_tokenize(sent)
    corefIndex, PRIndex = getPRPCorefIndex(coref)
    # print("corefIndex", corefIndex)

    FixedTokens = set()
    for item in depTree[1:]: # the first one is the root
        relation, fromID, toID = item
        fromID, toID = fromID - 1, toID - 1
        # print("fromID, toID", fromID, toID)
        if toID in PRIndex or fromID in PRIndex:
            if relation in ('nsubj', 'amod'): # consider these two dependencies
                FixedTokens.add(tokens[fromID])
                FixedTokens.add(tokens[toID])
    FixedTokens |= {tokens[i] for i in corefIndex}
    
    return FixedTokens


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
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    synonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, synonyms))
    antonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, antonyms))
    
    return set(synonyms) | set(antonyms) - {word}


def bertInit():
    berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
    bertori = BertModel.from_pretrained("bert-large-cased")
    bertmodel.eval()
    bertori.eval()

    return bertmodel, berttokenizer, bertori

bertmodel, berttoken, bertori = bertInit()

def get_perplexity(sentence):
    # Define the sentence to calculate perplexity for
    # sentence = ' '.join(tokens) 

    # Tokenize the sentence and convert to tensor
    tokens = berttoken.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # print(input_ids.shape)


    # Calculate the log probabilities of the words in the sentence
    outputs = bertmodel(input_ids, labels=input_ids)
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    # print(input_ids[:, 1:].shape)
    # print(input_ids[:, 1:].unsqueeze(-1).shape)
    # print(log_probs[0, :-1, :].shape)

    word_log_probs = log_probs[0, :-1, :].gather(1, input_ids[:, 1:]).squeeze(-1)

    # Calculate the perplexity of the sentence
    perplexity = torch.exp(-word_log_probs.mean())
    return round(perplexity.item(), 2)

def BertM(bert, berttoken, inpori, bertori, FixedTokens, pos_inf):
    global lcache
    for k in lcache:
        if inpori == k[0]:
            return k[1], k[2]
    sentence = inpori
    tokens = berttoken.tokenize(sentence)
    
    batchsize = 1000 // len(tokens)
    gen = []
    ltokens = ["[CLS]"] + tokens + ["[SEP]"]
    try:
        encoding = [berttoken.convert_tokens_to_ids(ltokens[0:i] + ["[MASK]"] + ltokens[i + 1:]) for i in
                    range(1, len(ltokens) - 1)] #.cuda()
    except:
        return " ".join(tokens), gen
    p = []
    for i in range(0, len(encoding), batchsize):
        tensor = torch.tensor(encoding[i: min(len(encoding), i + batchsize)]) #.cuda()
        pre = F.softmax(bert(tensor)[0], dim=-1).data.cpu()
        p.append(pre)
    pre = torch.cat(p, 0)
    # print (len(pre), len())
    # topk = torch.topk(pre[i][i + 1], K_Number)#.tolist()
    tarl = [[tokens, -1]]
    replaced = []
    
    # iterate every token in the sentence, finding words to be replaced
    for i in range(len(tokens)):
        if tokens[i] in string.punctuation:
            continue

        topk = torch.topk(pre[i][i + 1], K_Number)  # .tolist()
        value = topk[0].numpy()
        topk = topk[1].numpy().tolist()

        # print (topk)
        topkTokens = berttoken.convert_ids_to_tokens(topk)
        # print (topkTokens)
        # DA = oriencoding[i]
        
        topkTokens = [x for x in set(topkTokens) - pps]
        
        # Workaround: if two tokenizers result in a same tokenization list, then apply 
        # synonyms and antonyms replacement using wordnet
        if len(tokens) == len(pos_inf):
            print("Apply synonyms and antonyms using wordnet!")
            print("topkTokens before: ", len(topkTokens))
            
            syn_ant_set = get_syn_ant(tokens[i], pos_inf[i][-1])
            topkTokens = [x for x in set(topkTokens) | syn_ant_set]
            
            # print("topkTokens after: ", len(topkTokens))
            # print()

        # tarl = []
        for index in range(len(topkTokens)):
            if value[index] < 0.1:
                break
            tt = topkTokens[index]
            # print (tt)
            if tt in string.punctuation:
                continue
            if tt.strip() == tokens[i].strip():
                continue
            
            # do not replace tokens that equals to the words in the coref or related.
            if not ((tokens[i].strip() in FixedTokens) or (tt.strip() in FixedTokens)):
                l = deepcopy(tokens)
                l[i] = tt
                tarl.append([l, i, value[index]])
                replaced.append((tokens[i].strip(), tt.strip()))
            else:
                continue

    if len(tarl) == 0:
        return " ".join(tokens), gen

    # print('tarl: ', tarl)
    # for followup_tokens, _, _ in tarl[1:]:
    #     perplexity = get_perplexity(followup_tokens)
    #     print('sentence = ', ' '.join(followup_tokens))
    #     print('perplexity = ', perplexity)
        
    
    lDB = []
    # batchsize = 100

    # oriencoding = bertori(torch.tensor([berttoken.convert_tokens_to_ids(ltokens)]).cuda())[0][0].Output.cpu().numpy()
    # oriencoding = bertori(torch.tensor([berttoken.convert_tokens_to_ids(ltokens)]).cuda())[0][0].Output.cpu().numpy()
    for i in range(0, len(tarl), batchsize):
        # tarlist = tarl[i: min(len(tarl), i + 300]
        # lDB.append(bertori(torch.tensor([berttoken.convert_tokens_to_ids(["[CLS]"] + l[0] + ["[SEP]"]) for l in tarl[i: min(i + batchsize, len(tarl))]]).cuda())[0].Output.cpu().numpy())
        lDB.append(bertori(torch.tensor([berttoken.convert_tokens_to_ids(["[CLS]"] + l[0] + ["[SEP]"]) for l in
                                         tarl[i: min(i + batchsize, len(tarl))]]))[0].data.cpu().numpy())

    lDB = np.concatenate(lDB, axis=0)

    # print ("-----------------")
    # print (len(lDB))
    # print (len(tarl))
    lDA = lDB[0]
    assert len(lDB) == len(tarl)
    assert len(tarl) == len(replaced) + 1
    tarl = tarl[1:]
    lDB = lDB[1:]
    for t in range(len(lDB)):
        sen = " ".join(tarl[t][0]).replace(" ##", "")  # + "\t!@#$%^& " + str(math.exp(value[index]))#.replace(" ##", "")

        perplexity = get_perplexity(sen)
        
        word_tokens = []
        for token in tarl[t][0]:
            if token.startswith('##'):
                word_tokens[-1] = word_tokens[-1] + token[:2]
            else:
                word_tokens.append(token)

        gen.append([perplexity, sen, word_tokens, replaced[t]])
    if len(lcache) > 4:
        lcache = lcache[1:]

    lcache.append([inpori, " ".join(tokens), gen])
    return " ".join(tokens), gen  # .replace(" ##", ""), gen


def generate(oriTokens, oriSent, coref=[[]]):
    pos_inf = nltk.tag.pos_tag(oriTokens)
    FixedTokens = getCorefRelatedIndex(oriSent, coref, pos_inf)
    
    _, gen = BertM(bertmodel, berttoken, oriSent, bertori, FixedTokens, pos_inf)
    gen = sorted(gen)[::-1]
    if len(gen) == 0:
        return [], [], []
    else:
        gen = gen[:Max_Mutants]
        perplexitys = [g[0] for g in gen]
        generated_sents = [formalize_format(g[1]) for g in gen]
        generated_tokens = [g[2] for g in gen]
        replaced_pairs = [g[3] for g in gen]
    return generated_sents, generated_tokens, replaced_pairs, perplexitys


def formalize_format(sent):
    # this formalization is to remove extra space generated by CAT.
    sent = sent.replace(' .', '.')
    sent = sent.replace(' ,', ',')
    sent = re.sub(r'(\d)\s.\s(\d)', r'\1.\2', sent)
    sent = re.sub(r'(\d)\s,\s(\d)', r'\1,\2', sent)
    return sent


def simple_test():
    sents = [('Mary likes the meal she made.', [[[0,0],[4,4]]]),
            #  ('The fish eats the worm because it is hungry.',[[[0,1],[6,6]]]),
            #  ('The fish eats the worm because it is tasty.', [[[3,4],[6,6]]])
             ]
    for test_sent, coref in sents:
        generated_sents, generated_tokens, replaced_pairs, perplexitys = generate(test_sent.split(' '), test_sent, coref)
        print('origin:', test_sent)
        print('{} follow-up is/are generated'.format(len(generated_sents)))
        print()
        print('\n'.join(generated_sents))
        print('Perplexity: ', perplexitys)
        for pair in replaced_pairs:
            print('{} -> {}'.format(pair[0], pair[1]))
        print()

# if __name__ == '__main__':
    # simple_test()