"""
This code is adapted from CAT (ICSE'2022) under CAT/NewThres/bertMuN.py
We use the generation part as a baseline
"""

import numpy as np
import string
import torch
import torch.nn.functional as F
from copy import deepcopy
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM

# from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

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


K_Number = 100
Max_Mutants = 20
lcache = []


def bertInit():
    berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
    bertori = BertModel.from_pretrained("bert-large-cased")
    bertmodel.eval()
    bertori.eval()

    return bertmodel, berttokenizer, bertori


def BertM(bert, berttoken, inpori, bertori):
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
                    range(1, len(ltokens) - 1)]  # .cuda()
    except:
        return " ".join(tokens), gen
    p = []
    for i in range(0, len(encoding), batchsize):
        tensor = torch.tensor(encoding[i: min(len(encoding), i + batchsize)])  # .cuda()
        pre = F.softmax(bert(tensor)[0], dim=-1).data.cpu()
        p.append(pre)
    pre = torch.cat(p, 0)
    # print (len(pre), len())
    # topk = torch.topk(pre[i][i + 1], K_Number)#.tolist()
    tarl = [[tokens, -1]]
    replaced = []
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

        # tarl = []
        for index in range(len(topkTokens)):
            if value[index] < 0.05:
                break
            tt = topkTokens[index]
            # print (tt)
            if tt in string.punctuation:
                continue
            if tt.strip() == tokens[i].strip():
                continue
            l = deepcopy(tokens)
            l[i] = tt
            tarl.append([l, i, value[index]])
            replaced.append((tokens[i].strip(), tt.strip()))

    if len(tarl) == 0:
        return " ".join(tokens), gen

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
        DB = lDB[t][tarl[t][1]]
        DA = lDA[tarl[t][1]]
        #        assert np.shape(DA) == np.shape(DB)
        cossim = np.sum(DA * DB) / (np.sqrt(np.sum(DA * DA)) * np.sqrt(np.sum(DB * DB)))
        #        print (cossim)
        # print ()
        if cossim < 0.85:
            continue
            # print ("------")
            # print (" ".join(oritokens))
            # print (" ".join(tokens))
            # print (" ".join(l))
        # Because it uses subword-level tokenizer to tokenize, so we process to get a sentense without subwords
        # For example, ['under', '##dog']  -> 'underdog'
        sen = " ".join(tarl[t][0]).replace(" ##", "")  # + "\t!@#$%^& " + str(math.exp(value[index]))#.replace(" ##", "")

        # Because it uses subword-level tokenizer to tokenize, so we need to change to word-level tokenization
        # For example, ['under', '##dog']  -> ['under', 'dog']
        word_tokens = []
        for token in tarl[t][0]:
            if token.startswith('##'):
                word_tokens[-1] = word_tokens[-1] + token[:2]
            else:
                word_tokens.append(token)

        # if check_tree(tag, sen):
        gen.append([cossim, sen, word_tokens, replaced[t]])
    if len(lcache) > 4:
        lcache = lcache[1:]

    lcache.append([inpori, " ".join(tokens), gen])
    return " ".join(tokens), gen  # .replace(" ##", ""), gen


def generate(sent, max_num=Max_Mutants):
    bertmodel, berttoken, bertori = bertInit()
    _, gen = BertM(bertmodel, berttoken, sent, bertori)
    gen = sorted(gen)[::-1]
    if len(gen) == 0:
        return [], [], []
    else:
        gen = gen[:max_num]
        generated_sents = [g[1] for g in gen]
        generated_tokens = [g[2] for g in gen]
        replaced_pairs = [g[3] for g in gen]

    return generated_sents, generated_tokens, replaced_pairs


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
    # plain_text = open('./baselines/output/CATOutput.txt', 'w')
    #
    # cnt = 0
    # for doc_id, oriTokens, oriSent, goldCoref in oriSentencePairs:
    #     generated_sents = generate(oriSent)
    #     gen_dict[oriSent] = generated_sents
    #
    #     plain_text.write(oriSent + '\n')
    #     plain_text.write('\n'.join(generated_sents) + '\n')
    #
    #     print(f'Finish Sentence {cnt}, generated {len(generated_sents)}.')
    #     cnt += 1
    #
    # plain_text.close()
    #
    # with open('./baselines/output/CATGen.dat', 'wb') as f:
    #     pickle.dump(gen_dict, f)
