from typing import List
import random


def generate_monthly_probs_noisy():
    # Start from january
    monthly_probs = [0.70, 0.65, 0.60, 0.50, 0.40, 0.35,
                     0.25, 0.25, 0.35, 0.50, 0.65, 0.70]
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    p = []
    for month_idx, length in enumerate(days_per_month):
        base = monthly_probs[month_idx]
        for _ in range(length):
            noise = random.uniform(-0.05, 0.05)
            val = min(max(base + noise, 0.0), 1.0)
            p.append(val)
    return p

def prob_rain_more_than_n ( p: List[float], n: int ) -> float:
    days = len(p)
    dp_prev = [0] * (days+1)
    dp_curr = [0] * (days+1)

    # Probability of having zero rainy days in 0 days
    dp_prev[0] = 1

    for i in range(1, days+1):
        for j in range(i+1):
            if j == 0:
                dp_curr[j] =(1 - p[i - 1]) * dp_prev[j]
            else:
                dp_curr[j] = ( 1 - p[i-1] ) * dp_prev[j] + p[i-1] * dp_prev[j-1]

        dp_prev, dp_curr = dp_curr, dp_prev
    p_all = sum(dp_prev[i] for i in range (n, days+1))
    return 1 if p_all > 1 else p_all

if __name__ == '__main__':
    random.seed(10)
    # epsilon = 1e-10
    # p = [random.uniform(epsilon, 1 - epsilon) for _ in range(365)]
    p = generate_monthly_probs_noisy()
    for test_n in [50, 90, 100, 120, 200, 300]:
        val = prob_rain_more_than_n(p, test_n)
        print(f'Probability of raining more than {test_n} days is {val}')
