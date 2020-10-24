import string
import re
import unicodedata
from bs4 import BeautifulSoup

def _text_cleaner(text):
    # Removes HTML tags
    text = BeautifulSoup(text, features="lxml").get_text()
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    # Remove api strings
    expressions = ['… [+ [0-9]*chars]$', '…']
    for i in expressions:
        text = re.sub(i, '', text)
    return text

# TODO: detect arabic and other languages. or if not english detector
# TODO: remove url links
def _detect_cjk(text):
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return True
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return True
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return True
    return False


def results_cleaner(agg_results):
    remove_idx = []
    # Go over results: detect articles with cjk characters and apply cleaning function
    for i, r in enumerate(agg_results):
        if _detect_cjk(r['text']):
            remove_idx.append(i)
            continue
        r['text'] = _text_cleaner(r['text'])
    # Remove articles with cjk characters
    for ix in sorted(remove_idx, reverse=True):
        del agg_results[ix]
    return agg_results


def join_results(results_list, test_size=0.2):
    from sklearn.model_selection import train_test_split
    # Go over results_list and remove duplicates based on 'text' 
    holder = {}
    for result in results_list:
        value = holder.setdefault(result['text'], [])
        value.append(result['_id'])
        value.append(result['col'])
        value.append(result['category'])
    join_results_list = [{'id': v[0], 'col': v[1], 'category': v[2], 'text': k} for k, v in holder.items()]
    # Split articles into train and test
    join_results_list_train, join_results_list_test = train_test_split(join_results_list, test_size=test_size)
    return join_results_list_train, join_results_list_test