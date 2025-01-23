To solve this question fist we convert the word dictionary to prefix trie. 
In this way we can easily find the words that start with a given phoneme. 
Also the trie would take at most as much space as the dictionary in case of no overlap between words . 
Then we can use a recursive function to find all possible combinations of words that can be formed by the given phonemes. 
In each end node of the trie we save all the words with that pronunciation in a list. 

To search for matching phonems we look if a root from
