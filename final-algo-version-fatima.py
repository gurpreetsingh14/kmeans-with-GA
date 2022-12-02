import numpy
import matplotlib.pyplot

# ARTIFICIAL DATA GENERATION

cluster1_num_samples = 10
cluster1_x1_start = 0
cluster1_x1_end = 5
cluster1_x2_start = 2
cluster1_x2_end = 6
cluster1_x1 = numpy.random.random(size=(cluster1_num_samples))
cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
cluster1_x2 = numpy.random.random(size=(cluster1_num_samples))
cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

cluster2_num_samples = 10
cluster2_x1_start = 4
cluster2_x1_end = 12
cluster2_x2_start = 14
cluster2_x2_end = 18
cluster2_x1 = numpy.random.random(size=(cluster2_num_samples))
cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
cluster2_x2 = numpy.random.random(size=(cluster2_num_samples))
cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

c1 = numpy.array([cluster1_x1, cluster1_x2]).T
c2 = numpy.array([cluster2_x1, cluster2_x2]).T

data = numpy.concatenate((c1, c2), axis=0)

#data
# the numpy lists 1st 10 list are X,Y coordinates of cluster 1 points and rest 10 list are 
# X,Y coordinates of cluster 2 points

num_clusters = 2

'''
euclidean_distance() accepts 2 inputs X and Y. 
One of these inputs can be a 2-D array with multiple samples, and the other input should be a 1-D array with just 1 sample. 
The function calculates and returns the Euclidean distances between each sample in the 2-D array and 
the single sample in the 1-D array
'''
def euclidean_distance(X, Y):
    return numpy.sqrt(numpy.sum(numpy.power(X - Y, 2), axis=1))

'''
Input: 
1. solution is the initial coordinates of centriods. For example,
if num_of_clusters = 2, then
solution = [C1-x, C1-y, C2-x, C2-y]
2. data is the whole data, all the samples
'''

def cluster_data(solution):
    global num_clusters, data
    feature_vector_length = data.shape[1]
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(data, cluster_centers[clust_idx])
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

    return cluster_centers, all_clusters_dists, clusters, clusters_sum_dist

'''
fitness_func() is created and calls the cluster_data() function and 
calculates the sum of distances in all clusters
'''
def fitness_func(solution):
    fit_list = []
    m,_ = solution.shape
    for t in range(m):
        _, _, _, clusters_sum_dist = cluster_data(solution[t])
        fitness = 1.0 / (numpy.sum(clusters_sum_dist) + 0.00000001)
        fit_list.append(fitness)

    return numpy.array(fit_list)

#GENERATE Initial coordinates for cluster center

def init_cluster_center(num_clusters,start_coord,end_coord):
    io = []
    rc = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    cluster1_num = num_clusters*2
    cluster1_x1_start = start_coord
    cluster1_x1_end = end_coord
    cluster1_x1 = numpy.random.random(size=(cluster1_num))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x1 = numpy.array([2.75, 3.5, 9.25, 17])
    io.append(cluster1_x1)
    for y in range(len(rc)):
        ui = rc[y]*cluster1_x1
        io.append(ui)
    return numpy.array(io)
# return: numpy array: [[C1-x C1-y C2-x C2-y][][][]....]

#SELECTION FUNCTION

def selection(pop,sample_size, fitness):
    m,n = pop.shape
    new_pop = pop.copy()
        
    for i in range(m):
        rand_id = numpy.random.choice(m, size=max(1, int(sample_size*m)), replace=False)
        max_id = rand_id[fitness[rand_id].argmax()]
        new_pop[i] = pop[max_id].copy()
    
    return new_pop

#CROSSOVER

def crossover(pop, pc):
    m,n = pop.shape
    # print("\n\n")
    # print(m)
    # print(n)
    new_pop = pop.copy()
    
    for i in range(0, m-1, 2):
        if numpy.random.uniform(0, 1) < pc:
            # print("\nhitting the if clause")
            # print("\nvalue of i")
            # print(i)
            pos = numpy.random.randint(0, n-1)
            # print("\npos")
            # print(pos)
            # print("before\n")
            # print(new_pop)
            first_value = new_pop[i, pos+1:].copy()
            second_value = pop[i+1, pos+1:].copy()
            new_pop[i, pos+1:] = second_value
            new_pop[i+1, pos+1:] = first_value
            # print("\nafter the switch")
            # print(new_pop)
            # if (pop == new_pop).all():
                # print("SAME")
            # else:
                # print("DIFFERENT")
            
            
            
    return new_pop

#MUTATION

def mutation(pop, pm):
    m,n = pop.shape
    new_pop = pop.copy()
    mutation_prob = (numpy.random.uniform(0, 1, size=(m,n)) < pm).astype(int)
    # print("\nmutation prob", mutation_prob)
    return (mutation_prob + new_pop)

#GENETIC ALGORITHM

def GeneticAlgorithm(func, num_clusters,start_coord,end_coord, 
                     ps=0.2, pc=1.0, pm=0.1, max_iter=100, random_state=1234):
    
    numpy.random.seed(random_state)
    pop = init_cluster_center(num_clusters,start_coord,end_coord)
    print("initial pop")
    print(pop)
    fitness = func(pop)
    best = [fitness.max()]
    print("initial best fitness", best)
    fitness_res = []
    i = 0
    #print('==============Population before SELECTION')
    #print(pop)
    #print('==============Fitness list')
    #print(fitness)
    while i < max_iter:
        pop = selection(pop, ps, fitness)
        # print('==============Population after SELECTION')
        # print(pop)
        pop = crossover(pop, pc)
        # print('==============Population after CROSSOVER')
        # print(pop)
        pop = mutation(pop, pm)
        # print('==============Population after MUTATION')
        # print(pop)
        fitness = func(pop)
        # print("current fitness", fitness)
        # print("\nfitness")
        # print(fitness)
        if fitness.max()>best[0]:
            fitness_res.append(fitness)
        i += 1
    
    #print(f'Generation {i} fitness {fitness:0.4f}')    
    #print(f'Best fitness: {max(fitness_res):0.4f} at generation {fitness_res.index(max(fitness_res))}')
            
    return fitness_res, best, pop

f,b,p = GeneticAlgorithm(fitness_func,2,0,20)
fitness = fitness_func(p)
best = [fitness.max()]
index = numpy.where(numpy.isclose(fitness, best))
print(p[index])
# numpy.where(numpy.isclose(p, best))
# print(f'fitness is: {f}')
print(f'best fitness is: {best}')
print(f'population of centroids is: {p}')
