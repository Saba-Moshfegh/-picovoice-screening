# -picovoice-screening

## Overview
In this repo I provided answer to 3 screening questions of Picovoise.
Each questions' folder contains codes and brief technical explanations
of solutions. 

## Links
- The probability of rain on a given calendar day in Vancouver is p[i], where i is the day's index. For
example, p[0] is the probability of rain on January 1st, and p[10] is the probability of precipitation on January 11th. Assume
the year has 365 days (i.e., p has 365 elements). What is the chance it rains more than n (e.g., 100) days in Vancouver?
Write a function that accepts p (probabilities of rain on a given calendar day) and n as input arguments and returns the
possibility of raining at least n days.
```python
def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
pass
```
[Question 1 Explanation](Q1/explanation.md)

- [Question 2 Explanation](Q2/explanation.md)
- [Question 4 Explanation](Q4/explanation.md)
