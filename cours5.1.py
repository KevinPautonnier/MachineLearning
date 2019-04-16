"""
    Premier exercice de notre cinquième cours de machine learning.
    Le but était d'utiliser surprise pour prédire la note que donnerai un
    utilisateur spécifique pour un film en fonction de ses notes sur d'autres films.
    Surprise nous fournis les données nécessaires.
"""

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    """
        ...
    """
    
    #get data from surprise
    data = Dataset.load_builtin('ml-100k')

    trainset, testset = train_test_split(data, test_size=.25)

    algo = SVD()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    #calculate the delta
    x = [elem[2] - elem[3] for elem in predictions]

    #number of column in the graph
    clmnNb = 69

    plt.hist(x, clmnNb, facecolor='b', alpha=0.75)

    plt.xlabel('Delta values')
    plt.ylabel('Number of same delta')
    plt.title('Delta of rating')

    plt.show()

if __name__ == "__main__":
    main()
