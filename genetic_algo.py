import numpy

num_clusters = 2
def init(m, n):
    return numpy.random.random(size=(m,n))

def euclidean_distance(X, Y):
    return numpy.sqrt(numpy.sum(numpy.power(X - Y, 2), axis=1))

def cluster_data(solution):
    # global num_cluster, data
    feature_vector_length = solution.shape[1]
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(solution, cluster_centers[clust_idx])
        all_clusters_dists.append(numpy.array(cluster_center_dists))

    cluster_centers = numpy.array(cluster_centers)
    all_clusters_dists = numpy.array(all_clusters_dists)

    cluster_indices = numpy.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_clusters):
        clusters.append(numpy.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(numpy.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = numpy.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist

def fitness_func(solution):
    _, _, _, _, clusters_sum_dist = cluster_data(solution)

    fitness = 1.0 / (numpy.sum(clusters_sum_dist) + 0.00000001)

    return fitness



def print_result(gen_num, pop, fitness, x):
    m = pop.shape[0]
    print(f'Generation {gen_num} max fitness {fitness.max():0.4f} at x = {x[fitness.argmax()]:0.4f}')
    print(f'Average fitness: {fitness.mean():0.4f}')

# this is the fitness function that needs to be adjusted to the problem.
def func(x):
    return None
    # return numpy.array([fitness_func(x) for xi in x])


def selection(pop, sample_size, fitness):
    m,n = pop.shape
    new_pop = pop.copy()
        
    for i in range(m):
        rand_id = numpy.random.choice(m, size=max(1, int(sample_size*m)), replace=False)
        max_id = rand_id[fitness[rand_id].argmax()]
        print(fitness[rand_id].argmax())
        print(rand_id[fitness[rand_id].argmax()])
        new_pop[i] = pop[max_id].copy()
    
    return new_pop

def crossover(pop, pc):
    m,n = pop.shape
    new_pop = pop.copy()
    
    for i in range(0, m-1, 2):
        if numpy.random.uniform(0, 1) < pc:
            pos = numpy.random.randint(0, n-1)
            new_pop[i, pos+1:] = pop[i+1, pos+1:].copy()
            new_pop[i+1, pos+1:] = pop[i, pos+1:].copy()
            
    return new_pop

def mutation(pop, pm):
    m,n = pop.shape
    new_pop = pop.copy()
    mutation_prob = (numpy.random.uniform(0, 1, size=(m,n)) < pm).astype(int)
    return (mutation_prob + new_pop) % 2


def decode(ss, a, b):
    n = ss.shape[1]
    x = []
    for s in ss:
        bin_to_int = numpy.array([int(j) << i for i,j in enumerate(s[::-1])]).sum()
        int_to_x = a + bin_to_int * (b - a) / (2**n - 1)
        x.append(int_to_x)
    return numpy.array(x)

def GeneticAlgorithm(func, pop_size, str_size, low, high, 
                     ps=0.2, pc=1.0, pm=0.1, max_iter=1000, eps=1e-5, random_state=None):
    
    numpy.random.seed(random_state)
    pop = init(pop_size, str_size) # generates the population
    x = decode(pop, low, high)
    fitness = func(x)
    best = [fitness.max()]
    print_result(1, pop, fitness, x)
    
    i = 0
    while i < max_iter and abs(best[-1]) > eps:
        pop = selection(pop, ps, fitness)
        pop = crossover(pop, pc)
        pop = mutation(pop, pm)
        x = decode(pop, low, high)
        fitness = func(x)
        best.append(fitness.max())
        i += 1
    
    print_result(i, pop, fitness, x)
    
    if i == max_iter:
        print(i, 'maximum iteration reached!')
        print('Solution not found. Try increasing max_iter for better result.')
    else:
        print('Solution found at iteration', i)
        
    return fitness, x, best, i, pop_size


fs, xs, best, i, m = GeneticAlgorithm(func, pop_size=1, str_size=20, low=1, high=3, random_state=69)
