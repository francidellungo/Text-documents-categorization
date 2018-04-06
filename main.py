from News20 import News20_classification
from reuters import Reuters_classification


def main():
    News20_classification('bayes')
    News20_classification('perceptron')
    Reuters_classification('bayes')
    Reuters_classification('perceptron')

main()