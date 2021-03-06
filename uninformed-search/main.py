import numpy as np
import queue


STATES = {"NV": 0, "FRONTIER": 1, "VIS": 2}

class Node():
    def __init__(self, id, label=None ) -> None:
        self.state = STATES["NV"]
        self.dist = -1
        self.distFinal = -1
        self.links = []
        self.parent = None
        self.id = id
        self.label = label

class Graph():
    def __init__(self, n_vertex, links=None, head=None) -> None:
        self.adj = [ Node(i) for i in range(n_vertex)]
        self.head = head
        self.time = 0   

    def get(self, id):
        return self.adj[id]

    def addLink(self, v1, v2):
        a = self.adj[v1] if v1 < len(self.adj) else Node(v1)
        b = self.adj[v2] if v2 < len(self.adj) else Node(v2)

        a.links.append(b)
        b.links.append(a)
    
    def resetStates(self):
        self.time = 0
        for el in self.adj:
            el.state = STATES['NV']
            el.dist = -1
            el.parent = None

    def print(self):
        for el in self.adj:
            print(f'<NODE>: {el.id}\n <LINKS>:')
            
            for link in el.links:
                print(link.id)
            print('------')
    
    def setLabels(self, labels):
        for el, label in zip(self.adj, labels):
            el.label = label

    def printBFSPath(self, dest):
        source = self.adj[dest]
        if source.parent is None:
            raise Exception('No path was found')
        
        path = []
        path.append(source.id) # starting node
        while source.id != self.head:
            source = self.adj[source.parent.id]
            path.append(source.id)
        path.reverse()
        
        s = ''
        for it in path:
            s += self.adj[it].label + '-> '
        
        print(f'Path found: {s[:-3]}')

    def dls(self, source, dest):
        self.resetStates()

        for i in range(len(self.adj)):
            print('searching in level: ', i)
            for child in self.adj[source].links:
                self.resetStates()
                child.parent = self.adj[source]
                self.dls_visit(child, self.adj[dest], i)
                if self.adj[dest].parent is not None:
                    print('Solution found!')
                    self.printBFSPath(dest)
                    return
        print('No solution found!')

    def dls_visit(self, u, dest, limit):
        if u == dest:
            return u
        elif limit == 0:
            return None
        
        for v in u.links:
            v.parent = u
            self.dls_visit(v, dest, limit-1)
            

        
        

    def dfs(self, source, dest):
        self.resetStates()

        for u in self.adj:
            if u.state == STATES['NV']:
                self.visit(u)
        
        self.printBFSPath(dest)


    def visit(self, u):

        self.time +1
        u.dist = self.time
        u.state = STATES['FRONTIER']

        for v in u.links:
            if v.state == STATES['NV']:
                v.parent = u
                self.visit(v)
        u.state = STATES['VIS']
        self.time += 1
        u.distFinal = self.time
        

    def bfs(self, source, dest):
        self.resetStates()

        node = self.adj[source]
        node.dist=0
        
        #if node == final
        frontier = queue.Queue()
        frontier.put(node)

        while not frontier.empty():
            node = frontier.get()
            node.state = STATES["FRONTIER"]
            for l in node.links:
                if l.state == STATES['NV']:
                    l.state = STATES["FRONTIER"]
                    l.dist = node.dist+1
                    l.parent = node
                    frontier.put(l)
            node.state = STATES["VIS"]
        self.printBFSPath(dest)



g = Graph(8)
g.head = 1
g.setLabels(['Araraquara', 'BH', 'S??o Carlos', 'Ribeir??o Preto', 'S??o Paulo', 'Tr??s Lagoas', 'Campo Grande', 'Campinas'])


g.addLink(0,1)
g.addLink(0,2)
g.addLink(0,3)
g.addLink(1,2)
g.addLink(1,5)
g.addLink(1,4)
g.addLink(2,5)
g.addLink(2,3)
g.addLink(3,6)
g.addLink(4,7)
g.addLink(4,5)
g.addLink(5,6)
#g.print()

g.bfs(1, 6)
g.dfs(1, 6)
g.dls(1,6)