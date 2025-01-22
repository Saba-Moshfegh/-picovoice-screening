from phoneme_trie import Trie, trie_from_dict
from typing import List

class WordCombsWithPhonemes:

    def __init__(self, word_dict):
        self.trie = trie_from_dict(word_dict)


    def find_word_combos_with_pronunciation(self, phonemes: List[str]) -> List[List[str]]:

        return self._combine_words(self._find_all_words(phonemes))


    def _find_all_words(self, phonemes: List[str]) -> List[List[str]]:

        result = []
        def backtrack(start: int, end: int):
            if end == len(phonemes)+1:
                return
            word = self.trie.search_phoneme(phonemes[start:end])
            if word:
                result.append(word)
                backtrack(end, end+1)
            else:
                end += 1
                if end == len(phonemes)+1:
                    result.append(False)
                backtrack(start, end)

        backtrack(0, 1)
        if False in result:
            return []
        else:
            return result

    def _combine_words(self, all_words: List[List[str]]) -> List[List[str]] | None:
        all_combs = []
        if not all_words:
            print('There is no valid combination')
            return
        def backtrack(comb, index):
            if index == len(all_words):
                return all_combs.append(comb)
            for member in all_words[index]:
                backtrack(comb + [member], index+1)

        backtrack([], 0)

        return all_combs

if __name__ == '__main__':

    word_dict = {'ABACUS': [['AE', 'B', 'AH', 'K', 'AH', 'S']],
                 'BOOK': [['B', 'UH', 'K']],
                 'THEIR': [['DH', 'EH', 'R']],
                 'THERE': [['DH', 'EH', 'R']],
                 'TOMATO': [['T', 'AH', 'M', 'AA', 'T', 'OW'], ['T', 'AH', 'M', 'EY', 'T', 'OW']]}

    obj = WordCombsWithPhonemes(word_dict)

    obj.find_word_combos_with_pronunciation(['DH', 'EH', 'R', 'DH', 'EH'])
    assert( obj.find_word_combos_with_pronunciation(['DH', 'EH', 'R', 'DH', 'EH', 'r'])
               == [["THEIR", "THEIR"],["THEIR", "THERE"],["THERE", "THEIR"],["THERE", "THERE"]])
