'''
' Solution from Artificial inteligence course in Computer Science Doctoral Program of UFMS/FACOM
' Author: Professor Bruno
'''

from torch import true_divide


def generateSons(s):
    '''
    ' s: list with two position, for each jug
    '''

    listOfSons = list()

    # fill jug #1
    state = [7, s[1]]
    listOfSons.append(state)

    # fill jug #2
    state = [s[0], 5]
    listOfSons.append(state)

    #empty jug #1
    state = [0, s[1]]
    listOfSons.append(state)

    #empty jug #2
    state = [s[0], 0]
    listOfSons.append(state)

    # pour jug #1 in jug #2
    if s[0] >= 5-s[1]:
        state = [s[0]-(5-s[1]), 5]
    else:
        state = [0, s[1]+s[0]]

    listOfSons.append(state)


    # pour jug #2 in jug #1
    if s[1]<= 7-s[0]:
        state = [s[0]+s[1], 0]
    else:
        state = [s[0]+(7-s[0]), s[1]-(7-s[0])]

    listOfSons.append(state)

    return listOfSons

def isGoal(s):
    if s[0] == 4 or s[1] == 4:
        return True
    return False

def child2str(s):
    return ''.join([str(v) for v in s])


def bfs(start):
    candidates = [start]
    parents = dict()

    visited = [start]

    while(len(candidates)>0):
        parent = candidates[0]
        print('candidates: ', candidates)
        del candidates[0]
        print('Visited: ', parent)

        if isGoal(parent):
            res = []
            node = parent
            while node != start:
                res.append(node)
                node = parents[child2str(node)]
            res.append(start)
            res.reverse()
            print('Solution found: ', res)
            input()
        for child in generateSons(parent):
            if child not in visited:
                print('Inserted: ', child, parent)
                visited.append(child)
                parents[child2str(child)] = parent
                candidates.append(child)


bfs([0,0])