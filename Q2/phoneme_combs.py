from phoneme_trie import trie_from_dict
from typing import List

class WordCombsWithPhonemes:
    '''
    Class to find all possible ways to segment the given phoneme list into words found in the trie by using the trie
    from words dictionary.
    '''

    def __init__(self, word_dict):
        self.trie = trie_from_dict(word_dict)

    def find_word_combos_with_pronunciation(self, phonemes: List[str]) -> List[List[str]]:
        """
        Returns a list of all possible ways to segment the given phoneme list
        into words found in the trie. Note that all phonemes should be used and their order should be preserved.

        :param phonemes: List of phonemes

        :return : List of all possible segmentations
        """

        if not phonemes:
            return []

        def backtrack(start: int) -> List[List[str]]:

            if start == len(phonemes):
                return [[]]

            results = []

            for end in range(start + 1, len(phonemes) + 1):
                word_list = self.trie.search_phoneme(phonemes[start:end])
                if word_list:
                    suffix_segmentations = backtrack(end)
                    for w in word_list:
                        for seg in suffix_segmentations:
                            results.append([w] + seg)

            return results

        # Start backtracking from index 0
        return backtrack(0)

if __name__ == '__main__':

    word_dict = {'ABACUS': [['AE', 'B', 'AH', 'K', 'AH', 'S']],
                 'BOOK': [['B', 'UH', 'K']],
                 'THEIR': [['DH', 'EH', 'R']],
                 'THERE': [['DH', 'EH', 'R']],
                 'TOMATO': [['T', 'AH', 'M', 'AA', 'T', 'OW'], ['T', 'AH', 'M', 'EY', 'T', 'OW']],
                 'HOMEMADE': [['HH', 'OW', 'M', 'AH', 'M', 'EY', 'D']],
                 'HOME': [['HH', 'OW', 'M']],
                 'MADE': [['M', 'EY', 'D']],
                 'FIREMAN': [['F', 'AY', 'ER', 'M', 'AH', 'N']],
                 'FIRE': [['F', 'AY', 'ER']],
                 'MAN': [['M', 'AH', 'N']],
    }

    obj = WordCombsWithPhonemes(word_dict)

    # Test 1
    assert obj.find_word_combos_with_pronunciation([]) == []

    # Test 2
    assert obj.find_word_combos_with_pronunciation(['DH', 'EH', 'R', 'DH', 'EH']) == []

    # Test 3
    assert( obj.find_word_combos_with_pronunciation(['DH', 'EH', 'R', 'DH', 'EH', 'r'])
               == [["THEIR", "THEIR"],["THEIR", "THERE"],["THERE", "THEIR"],["THERE", "THERE"]])

    # Test 4
    assert (obj.find_word_combos_with_pronunciation(
        ['DH', 'EH', 'R', 'HH', 'OW', 'M', 'AH', 'M', 'EY', 'D'])
            == [["THEIR", "HOMEMADE"],["THERE", "HOMEMADE"]])

    # Test 5
    assert (obj.find_word_combos_with_pronunciation(
        ['DH', 'EH', 'R', 'F', 'AY', 'ER', 'M', 'AH', 'N'])
            == [["THEIR", "FIRE", "MAN"], ["THEIR", "FIREMAN"],["THERE", "FIRE", "MAN"], ["THERE", "FIREMAN"]])
