from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor

population_size = 10
max_generations = 5
crossover_rate = 0.8
mutation_rate = 0.05
C = 1.0  # Parametru SVM


def calc_term2(i, j, data, alpha, labels):
    """
    Functia care calculeaza termenul 2 al functiei de cost pentru optimizarea SVM
    """
    kernel_value = np.dot(data[i], data[j])
    return (alpha[i] * labels[i]) * (alpha[j] * labels[j]) * kernel_value


def compute_fitness(alpha, data, labels):
    """
    Calculeaza fitness-ul folosind operatii matriciale.
    Optimizat pentru performanța.
    """
    alpha = np.array(alpha)
    labels = np.array(labels)

    term1 = np.sum(alpha)

    n_samples = len(data)

    #paralelism deoarece programul crapa la memorie si pe langa numar redus de antrenare si calcul matriceal am pus si concurenta
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(n_samples):
            for j in range(i, n_samples):
                futures.append(executor.submit(calc_term2, i, j, data, alpha, labels))

        term2 = sum(f.result() for f in futures)

    return term1 - 0.5 * term2


def adjust_constraints(alpha, labels):
    """
    Functia care ajusteaza coeficientii Lagrange pentru a respecta constrângerile SVM
    """
    labels = labels.flatten()
    sum_constraints = np.sum(alpha * labels)
    adjustment = sum_constraints / len(labels)
    alpha -= labels * adjustment
    alpha = np.clip(alpha, 0, C)
    return alpha


def initialize_population(size, n_samples):
    """
    Functia care inițializeaza populatia
    """
    return [np.random.uniform(0, C, n_samples) for _ in range(size)]


def crossover(parent1, parent2):
    """
    Functia de crossover care combină doi părinti pentru a crea copii
    """
    point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(individual):
    """
    Functia care efectuează mutatia asupra unui individ
    """
    index = np.random.randint(0, len(individual))
    individual[index] += np.random.uniform(-0.5, 0.5)
    individual[index] = np.clip(individual[index], 0, C)
    return individual


def select_parent(population, fitness):
    """Functia de selectie a parintilor pe baza fitness-ului"""
    idx1, idx2 = np.random.choice(len(population), 2, replace=False)
    return population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]


def algoritm_evolutiv(data, labels):
    """
    Algoritmul evolutiv pentru optimizarea SVM.
    """
    population = initialize_population(population_size, len(data))

    for generation in range(max_generations):
        fitness = [compute_fitness(ind, data, labels) for ind in population]

        print(f"Generatia {generation + 1}/{max_generations}, cel mai bun fitness: {max(fitness)}")

        new_population = []

        for _ in range(population_size // 2):
            parent1 = select_parent(population, fitness)
            parent2 = select_parent(population, fitness)

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if np.random.rand() < mutation_rate:
                child1 = mutate(child1)
            if np.random.rand() < mutation_rate:
                child2 = mutate(child2)

            child1 = adjust_constraints(child1, labels)
            child2 = adjust_constraints(child2, labels)

            new_population.extend([child1, child2])

        population = new_population

    fitness = [compute_fitness(ind, data, labels) for ind in population]
    return population[np.argmax(fitness)]


def train_and_evaluate(data, labels, alpha):
    """Functia care antreneaza și evalueaza modelul SVM"""
    support_vectors = [(data[i], labels[i], alpha[i]) for i in range(len(alpha)) if alpha[i] > 1e-5]
    print(f"Numar vectori suport: {len(support_vectors)}")
    print("Antrenare completa.")


def reduce_dimension(x_train, n_components=5):
    """Functia care reduce dimensiunea setului de date folosind PCA(principal component analysis)"""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x_train)


def main():
    wine_quality = fetch_ucirepo(id=186)

    x = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy()

    # luam calitatea ca un set binar
    y = np.where(y > 8, -1, y) #calitate > n, eticheta = -1
    y = np.where((y <= 8) & (y > 3), 1, y) # m < calitate <= n, eticheta = 1
    y = np.where(y <= 3, 0, y) #calitate <= m, eticheta = 0

    scalar = StandardScaler() #functie de netezire pentru scalarea datelor
    x = scalar.fit_transform(x)

    x = reduce_dimension(x, n_components=5)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train_reduced, y_train_reduced = x_train[:30], y_train[:30]

    best_alpha = algoritm_evolutiv(x_train_reduced, y_train_reduced)

    train_and_evaluate(x_train_reduced, y_train_reduced, best_alpha)


if __name__ == "__main__":
    main()
