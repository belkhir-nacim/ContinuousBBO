def calculate(limit):
    primes = []
    divisor = 0
    for current in range(limit):
        previous = []
        for divisor in range(2, current):
            if current % divisor == 0:
                break
        if divisor == current - 1:
            primes.append(current)
    return primes
