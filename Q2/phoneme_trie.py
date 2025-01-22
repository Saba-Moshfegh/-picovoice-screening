from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        self.endWords = []

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert_word(self, phoneme:List[str], word:str):

        curr_node = self.root
        for ph in phoneme:
            ph = ph.lower()
            if ph in curr_node.children:
                curr_node = curr_node.children[ph]
            else:
                curr_node.children[ph] = TrieNode()
                curr_node = curr_node.children[ph]

        curr_node.endWords.append(word)

    def search_phoneme(self, phoneme:List[str]):

        curr_node = self.root
        for ph in phoneme:
            ph = ph.lower()
            if ph in curr_node.children:
                curr_node = curr_node.children[ph]
            else:
                return False

        return curr_node.endWords

def print_trie(node: TrieNode):

    if node.endWords:
        print(node.endWords)
    if not node:
        return
    for char, node in node.children.items():
        print_trie(node)

def trie_from_dict(word_phoneme_dict: dict):

    obj = Trie()

    for word, phonemes in word_phoneme_dict.items():
        for phoneme in phonemes:
            obj.insert_word(phoneme, word)

    return obj


if __name__ == '__main__':

    word_dict = {'ABACUS': [['AE', 'B', 'AH', 'K', 'AH' , 'S']],
    'BOOK': [['B', 'UH', 'K']],
    'THEIR': [[ 'DH', 'EH', 'R']],
    'THERE': [['DH', 'EH', 'R']],
    'TOMATO': [ ['T', 'AH', 'M', 'AA', 'T', 'OW'], ['T', 'AH', 'M', 'EY', 'T', 'OW']] }

    obj = trie_from_dict(word_dict)
    print_trie(obj.root)
    assert obj.search_phoneme(['T', 's' ,'k']) == False
    assert not obj.search_phoneme(['T', 'AH', 'M', 'EY', 'T', 'OW']) == False