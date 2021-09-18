
import pandas as pd, numpy as np
try:
	import bert
except:
	print("bert-for-tf2 not installed")

# transforms sentences to ids, masks and segment ids prepared to feed bert
def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')
    
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)
    
    return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=200):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    
    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    
    return all_input_ids, all_input_mask, all_segment_ids

def generate_data_for_tokenizer(split_text,target_series):
    labels_list = []
    dates = []
    for date, arrays in split_text.itertuples():
        dates.extend([date]* len(arrays))
    for date in dates:
        labels_list.append(target_series.loc[date])
    
    split_text_flat = split_text.values.flatten()
    sentence_list = [sentence for array in split_text_flat for sentence in array]
    
    labels = pd.DataFrame(labels_list, index = dates)
    sentences  = pd.DataFrame(sentence_list, index = dates)
    return sentences, labels

# given an input text and a set of keywords, returns the top_n_terms which contain any of the keywords by frequency of appearance.
def find_new_token_with_custom_keywords(array_of_text, custom_keywords, top_n_terms, extra_tokens):
    
    def contains_keyword(word,keywords):
        for k in keywords:
            if word.find(k) >= 0:
                return True
        return False
    
    def count_frequency(my_list): 
        freq = {} 
        for item in my_list: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
        return freq
    
    raw_text = "".join(array_of_text).replace(".com","-com").replace(".", "").replace(",", "").replace("\n", " ").replace("-com",".com")
    raw_words = raw_text.split(" ")
    matches = []
    for word in raw_words:
        if contains_keyword(word.lower(),custom_keywords):
            matches.append(word.lower())
    
    matches_count = count_frequency(matches)
    #sorts the counts
    #matches_dict = {k: v for k, v in sorted(matches_count.items(), key=lambda item: item[1], reverse = True)}
    # selects top n words from the list
    #new_tokens = list(matches_dict)[:top_n_terms]  + extra_tokens
    import operator
    sorted_x = sorted(matches_count.items(), key=operator.itemgetter(1), reverse = True)
    new_tokens = [ tup[0] for tup in sorted_x[:top_n_terms]]  + extra_tokens
    
    print("New tokens to be added: ",new_tokens)
    return new_tokens

# creates bert tokenizer
def create_tokenizer(vocab_file='vocab.txt', do_lower_case=True):
    return bert.bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# appends extra tokens to the vocab of the tokenizer
def add_new_tokens(new_vocab, tokenizer):
    for i in range(len(new_vocab)):
        new_key = new_vocab[i]
        old_key = "[unused{}]".format(i)
        value = tokenizer.vocab.pop(old_key)
        tokenizer.vocab[new_key] = value
    return tokenizer

# transforms bet output in one continuous series removing the padding
def bert_output_to_one_time_series_per_day(bert_inputs, bert_output, sentences):
    n_sentences = sentences.groupby(sentences.index).count()
    n_tokens = bert_inputs["input_mask"][:].sum(axis = 1)
    mask_out = [bert_output[1][counter,:length,:] for length, counter in zip(n_tokens,range(len(n_tokens)))]
    
    articles_per_day = []
    acc = 0
    for n in n_sentences.values:
        n = n[0]
        concat_articles = np.array(mask_out[acc:acc + n])
        flattened = []
        for sentence in concat_articles:
            for token in sentence:
                flattened.append(token)
        flattened = np.array(flattened)
        #flattened = np.array([token for token in sentence for sentence in concat_articles])
        articles_per_day.append(flattened)
        acc += n
    return np.array(articles_per_day)

# prepares_labels
def label_transformer(prices, mode = "returns", shift = 5, index = None, standarized = False):
    prices = pd.DataFrame(prices)
    prices.columns = ["today"]
    if index is not None:
        prices = prices[index]
    prices["tomorrow"] = prices.shift(1)
    prices["returns"] = prices["today"].pct_change()
    prices["diff"] = prices["today"] - prices["tomorrow"]
    def standard(df):
        return (df - df.mean())/df.std()
    output = prices[mode].shift(shift).dropna()
    return output if not standarized else standard(output)



def dummy():
	return "dudhduh"