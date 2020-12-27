import string, json


# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"



def get_emission_and_vocab():
    #get vocab
    with open("vocab.json", "r") as file1: 
        vocab = json.load(file1)
        
    #get emission_counts
    with open("emission_counts.json", "r") as file2: 
        emission = json.load(file2)
        
    emission_counts = {}
    
    for key in emission.keys():
        key_l = key.split(' ')
        new_key = (key_l[0], key_l[1])
        emission_counts[new_key] = emission[key]
        
    return vocab, emission_counts



def my_preprocess(vocab, sentence):
    """
    Preprocess data
    """
    punct = set(string.punctuation)
    
    orig = []
    prep = []

    # Read data
    file = sentence.split()
    new_file = []
    for word in file:
        if any(char in punct for char in word):
            for i in punct:
                s_l = word.split(i)
                if len(s_l) < 2:
                    continue
                else:
                    s_l[1] = i+s_l[1]
                    for j in s_l:
                        new_file.append(j)
        else:
            new_file.append(word)

    for cnt, word in enumerate(new_file):

        # End of sentence
        if not word.split():
            orig.append(word.strip())
            word = "--n--"
            prep.append(word)
            continue

        # Handle unknown words
        elif word.strip() not in vocab:
            orig.append(word.strip())
            word = assign_unk(word)
            prep.append(word)
            continue

        else:
            orig.append(word.strip())
            prep.append(word.strip())

    return orig, prep




def predict_pos(prep, emission_counts, vocab):
    '''
    Input: 
        prep: a preprocessed sentence to predict POS for
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output: 
        tags: a list of POS tags for prep
    '''
    
    states = ['#', '$', "''", '(', ')', ',', '--s--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
              'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB',
              'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD','VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
              'WP', 'WP$', 'WRB', '``']

    
    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())
    pos_list = []
    
    for word in prep:
        count_final = 0
        pos_final = ''
        
        if word in vocab:
            for pos in states:
                key = (pos, word)
                
                if key in emission_counts.keys():
                    count = emission_counts[key]
                    
                    if count > count_final:
                        count_final = count
                        pos_final = pos
            pos_list.append(pos_final)
            
    return pos_list
