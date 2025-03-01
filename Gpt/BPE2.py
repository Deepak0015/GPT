from collections import Counter , defaultdict
import re 


def get_initial_tokens(text):
    tokens = [list(word) + ['</w>'] for word in text.split()]
    return tokens

def count_pairs(corpus):
    pairs = defaultdict(int)
    for word in corpus:
        for i in range(len(word)- 1):
            pairs[(word[i],word[i+1])] +=1 
    return pairs


def merge_pair(pair, corpus):
    new_corpus = []
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in corpus:
        word_str = ' '.join(word)
        word_str = pattern.sub(''.join(pair), word_str)
        new_corpus.append(word_str.split())
    return new_corpus

def byte_pair_encoding(text, num_merges):
    corpus = get_initial_tokens(text)
    if not corpus:  # Handle edge case
        return [], []

    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        corpus = merge_pair(best_pair, corpus)
        merges.append(best_pair)
    return corpus, merges

corpus = '''Tokenization is the process of breaking down 
a sequence of text into smaller units called tokens,
which can be words, phrases, or even individual characters.
Tokenization is often the first step in natural languages processing tasks 
such as text classification, named entity recognition, and sentiment analysis.
The resulting tokens are typically used as input to further processing steps,
such as vectorization, where the tokens are converted
into numerical representations for machine learning models to use.'''

n = 230
bpe_pairs = byte_pair_encoding(corpus, n)
print(bpe_pairs[0])