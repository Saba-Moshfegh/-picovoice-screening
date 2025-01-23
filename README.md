# -picovoice-screening

## Overview
In this repo I provided answer to 3 screening questions of Picovoise.
Each questions' folder contains codes and brief technical explanations
of solutions. 

## Question 1
- The probability of rain on a given calendar day in Vancouver is p[i], where i is the day's index. For
example, p[0] is the probability of rain on January 1st, and p[10] is the probability of precipitation on January 11th. Assume
the year has 365 days (i.e., p has 365 elements). What is the chance it rains more than n (e.g., 100) days in Vancouver?
Write a function that accepts p (probabilities of rain on a given calendar day) and n as input arguments and returns the
possibility of raining at least n days.
```
def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
pass
```
[Question 1 Explanation](Q1/explanation.md)

- A phoneme is a sound unit (similar to a character for text). We have an extensive pronunciation
dictionary (think millions of words). Below is a snippet:

ABACUS AE B AH K AH S  
BOOK B UH K  
THEIR DH EH R  
THERE DH EH R  
TOMATO T AH M AA T OW  
TOMATO T AH M EY T OW  

Given a sequence of phonemes as input (e.g. ["DH","EH"
,"R", can produce this sequence (e.g. [["THEIR","THEIR"], ["THEIR","DH","EH","THERE"], ["THERE","R"]), find all the combinations of the words that
"THEIR"], ["THERE"
,"THERE"]]). You can preprocess the dictionary into a different data structure if needed.

```
def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
pass

```
[Question 2 Explanation](Q2/explanation.md)
- [Question 4 Explanation](Q4/explanation.md)
