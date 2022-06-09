from decimal import MIN_EMIN
from matplotlib import pyplot as plt
import numpy as np
import random
import math


MIN_VAL = 0
MAX_VAL = 200
N_SAMPLES = 300
K_SIZE = 3
COLORS = ['r','g','b']


def dist(x,y):
    return math.sqrt((x[0]-y[0])**2+ (x[1]-y[1])**2 )


def kmeans(cx, cy, X, Y, cid, max_it=50, max_it_no_changes=5):
    twc = 0 #time without changes
    it = 0
    last_changes = 0
    while it < max_it and twc<max_it_no_changes:
        changes = 0
        for i in range(N_SAMPLES):

            dists  = [dist((cx[j], cy[j]), (X[i], Y[i])) for j in range(K_SIZE)]
            
            nid    = dists.index(min(dists))

            changes += int(nid != cid[i])

            cid[i] = nid

        for id in range(K_SIZE):
            ids = np.where(cid==id)
            idx = X[ids]
            idy = Y[ids]

            cx[id] = np.mean(idx)
            cy[id] = np.mean(idy)

            plt.plot(idx, idy, COLORS[id]+'o', label='Cluster '+str(id))
            plt.plot(cx[id], cy[id], '+k', markersize=16)
        plt.title(f'Iteration {it}')
        plt.legend(loc='upper right')
        plt.savefig(f'result.png')
        plt.show()

        twc = twc+1 if changes == last_changes else 0

        print(f'Iteration {it} - Changes: {changes}')
        last_changes = changes
        it += 1



def main():
    X = np.random.randint(MIN_VAL, MAX_VAL, (N_SAMPLES))
    Y = np.random.randint(MIN_VAL, MAX_VAL, (N_SAMPLES))

    cx = np.random.randint(MIN_VAL, MAX_VAL, (K_SIZE))
    cy = np.random.randint(MIN_VAL, MAX_VAL, (K_SIZE))


    plt.plot(X, Y, 'ko')
    for i in range(K_SIZE):
        #plt.plot(cx, cy, 'go', markersize=12)
        plt.plot(cx[i], cy[i], COLORS[i]+'+', markersize=16, label='Cluster '+str(i))
    plt.title('Before running K-Means')
    plt.legend(loc='upper right')
    plt.show()

    cid = np.zeros((N_SAMPLES), dtype=int)

    kmeans(cx, cy, X, Y, cid)


if __name__ == '__main__':
    main()

