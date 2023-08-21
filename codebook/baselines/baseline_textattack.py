from textattack.transformations import WordSwapRandomCharacterDeletion, CompositeTransformation
from textattack.augmentation import EmbeddingAugmenter, Augmenter
from nltk.tokenize.treebank import TreebankWordTokenizer

from codebook.utils import diffStrings

tokenizer = TreebankWordTokenizer()
transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])


def generate(text, max_num=10):
    gens = []
    augmenter1 = Augmenter(transformation=transformation, transformations_per_example=int(max_num/2))
    gens.extend(augmenter1.augment(text))

    augmenter2 = EmbeddingAugmenter(transformations_per_example=max_num - int(max_num/2))
    gens.extend(augmenter2.augment(text))

    # remove duplicate
    gens = list(set(gens))

    # tokenize
    new_tokens = [tokenizer.tokenize(x) for x in gens]
    return gens, new_tokens, [diffStrings(text, new) for new in gens]


def simple_test():
    test_sent = 'But if not, your blessing of peace will come back to you .'
    generated_sents, generated_tokens, replaced_pairs = generate(test_sent)

    print(len(generated_sents))
    print('\n'.join(generated_sents))
    for pair in replaced_pairs:
        print('{} -> {}'.format(pair[0], pair[1]))


if __name__ == '__main__':
    simple_test()
