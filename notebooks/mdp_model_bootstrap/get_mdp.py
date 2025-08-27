import sys

x = float(sys.argv[1])


def p(alpha):
    return ((1 - x) / (1 + x)) ** (1 / alpha)


def mdp(alpha):
    return 5.5e-4 / (0.015 * p(alpha))


results = [(alpha, p(alpha), mdp(alpha)) for alpha in [0.1, 0.5, 1, 2, 5, 10]]
formatted_results = "\n".join(
    [
        f"{alpha}\t{100 * p(alpha):.2f}%\t{100 * mdp(alpha):.2f}%"
        for alpha in [0.1, 0.5, 1, 2, 5, 10]
    ]
)

print("alpha\tp(driver|f>0)\tMDP")
print(formatted_results)
