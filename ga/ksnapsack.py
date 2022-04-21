#Based on:
# https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/


from numpy.random import randint
from numpy.random import rand
from functools import reduce
import numpy as np


#dict with weight->benefit
weights = np.array([4, 5, 7, 9, 6])
benefits = np.array([2, 2, 3, 4, 4])
max_cap = 23
n_pop = 100
n_elem_cr = 5
n_bits_cr = 4
n_bits = n_elem_cr*n_bits_cr
max_iter = 100


def selection(population, scores, k=3):
    idx = randint(len(population))

    for i in randint(0, len(population), k-1):
        if scores[i] < scores[idx]:
            idx = i
    return population[idx]

def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()

    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        
        c1 = p1[:pt]+p2[pt:]
        c2 = p2[:pt]+p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1-bitstring[i]

def cr_to_int(cr):
    s = []
    for i,pos in enumerate(range(0,n_bits, n_bits_cr)):
        gene = cr[pos:pos+n_bits_cr]
        exp = list(range(0,4) )
        
        #print(type(exp[0]), type(gene[0]))

        size = reduce(lambda sum, x: sum+ x[0]*2**x[1], zip(gene,exp), 0)
        s.append(size)

    return s

def bag_sum(cr):
    s = cr_to_int(cr)

    s = np.array(s)
    
    return (sum(s*weights), sum(s*benefits))



def genetic_alg(fitness_fn, r_cross, r_mut):
    population = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    score = fitness_fn(population[0])
    if score[0] > max_cap:
        population[0] = list(np.zeros(20, dtype=int))
        score = (0.,0.)

    #print(population[0])
    #print(score)

    best_score = (population[0].copy(), score) # 0: índice, 10: peso, 11: benefício

    for gen in range(max_iter):
        scores = [fitness_fn(cr) for cr in population]

        for i, score in enumerate(scores):
            if score[0]< max_cap and score[1] > best_score[1][1]:
                best_score = (population[i].copy(), score)
                #print('new best score found at ', i, '-st element, score: ', best_score[1])

        selected = [selection(population, scores) for _ in range(n_pop)]
        #print(np.shape(selected))
        
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]

            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
            
        population = children
        if gen%5==0:
            print(f'Generation[{gen}], Best found [Weight: {best_score[1][0]}, Benefit: {best_score[1][1]}]')

    return best_score
            


solution = genetic_alg(bag_sum, 0.9, 1.0/float(n_bits))
print(f'Best found [Weight: {solution[1][0]}, Benefit: {solution[1][1]}]')
print(cr_to_int(solution[0]))