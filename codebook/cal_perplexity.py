from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
MAXTOKEN_LENGTH = 512


def bertInit():
    berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
    bertori = BertModel.from_pretrained("bert-large-cased")
    bertmodel.eval()
    bertori.eval()

    return bertmodel, berttokenizer, bertori

bertmodel, berttoken, bertori = bertInit()

def sample_maxGen(df, maxGen=20):
    df_all = None
    # print('OID = ', set(df['OID']))
    for oid in range(0, max(df['OID'])):
        if oid not in df['OID']:
            print('skip: oid = ', oid)
            continue
        df1 = df[df.OID == oid]
        if len(set(df1.pairConsistent.values)) == 2:
            dff = df1[df1.pairConsistent == False]
            if len(dff) >= maxGen:
                df1 = dff.iloc[:maxGen, :]
            else:
                dft = df1[df1.pairConsistent == True].iloc[:maxGen - len(dff), :]
                df1 = pd.concat([dff, dft], axis=0)
        else:
            df1 = df[df.OID == oid].iloc[:min(len(df1), maxGen), :]

        if df_all is None:
            df_all = df1
        else:
            df_all = pd.concat([df_all, df1], axis=0)
            # print(len(df_all))
        # print()
    return df_all

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


if __name__ == '__main__':
    for method in ('CAT', 'Checklist', 'Textattack'): # 'SIT', 'PatInv', , 'CrestNew'
        print(f'Method = {method}')
        
        df = pd.read_csv('../NewOutput_NeuralCoref/{}.tsv'.format(method), delimiter='\t')
        # df = pd.read_csv('../NewOutput_Stats/update_{}.tsv'.format(method), delimiter='\t')
        df_sampled = sample_maxGen(df, maxGen=20)
        
        oriPerList = []
        newPerList = []
        
        for i in tqdm(range(len(df_sampled))):
            oriSent = df_sampled.iloc[i, 4]
            genSent = df_sampled.iloc[i, 5]
            
            # truncate the sentence to avoid the situation:
            # `token indices sequence length is longer than the specified maximum sequence length'
            oriSent = oriSent[:min(len(oriSent), MAXTOKEN_LENGTH)]
            genSent = genSent[:min(len(genSent), MAXTOKEN_LENGTH)]
            
            oriPer = get_perplexity(oriSent)
            genPer = get_perplexity(genSent)
            
            oriPerList.append(oriPer)
            newPerList.append(genPer)
            
        assert len(oriPerList) == len(newPerList) == len(df_sampled)
        
        df_sampled['oriPerplexity'] = oriPerList
        df_sampled['newPerplexity'] = newPerList
        
        df.to_csv('../NewOutput_NeuralCoref/per_{}.tsv'.format(method), sep='\t')