import numpy as np
import time
import torch

#calculate top-k similarity function 
def structureSimi(input_dir, mode, k):
    all_paths = []
    name = mode + '_StructureSimilar_WQ_Path_' + str(k)
    with open(os.path.join(input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            edgeList, g_adj = getSubgraphInfor(line) # edgeList is a list, g_adj is a dict
            relationPaths = getSubgraphPath(edgeList, g_adj) # all_paths is a relation's list
            all_paths.append(relationPaths)
    if mode == 'train':
        n = len(all_paths) # all element number
        print("n:", n)        
        similarityMatrix = np.zeros((n, n)) # it is a n*n array
        similarDict = dict()
        for i in range(n):
            j = i + 1
            while i < n-1 and j <= n-1:
                similarityMatrix[i][j] = similarityPath(all_paths[i], all_paths[j])
                similarityMatrix[j][i] = similarityMatrix[i][j] 
                j += 1
            temp = []
            temp.append(heapq.nlargest(k, range(len(similarityMatrix[i])), similarityMatrix[i].__getitem__))#the top-k index, it is a list
            temp.append(heapq.nlargest(k, similarityMatrix[i])) # the top-k max similarity score, it is a list     
            similarDict[i] = temp
    elif mode == 'test':
        all_test_paths = []
        with open(os.path.join(input_dir, mode + '.json')) as ft:
            for line in ft.readlines():
                line = json.loads(line.strip())
                edgeList, g_adj = getSubgraphInfor(line) 
                relationPaths = getSubgraphPath(edgeList, g_adj) 
                all_test_paths.append(relationPaths)
        n = len(all_test_paths)
        print("n:", n)
        similarDict = dict()  
        for i in range(n):
            similar_temp = []
            for j in range(len(all_paths)):
                similar_temp.append(similarityPath(all_test_paths[i], all_paths[j]))
            temp = []
            temp.append(heapq.nlargest(k, range(len(similar_temp)), similar_temp.__getitem__))
            temp.append(heapq.nlargest(k, similar_temp))
            similarDict[i] = temp
    
    with open(os.path.join(input_dir, '{}.pt'.format(name)), 'wb') as fw:
        pickle.dump(similarDict, fw)
        
# return the subgraph's edgeList and adj            
def getSubgraphInfor(line):
    g_adj = line['inGraph']['g_adj']
    edgeList =[] #every element is a edge, the edge is a tuple
    for head, value in g_adj.items():
        for tail, _ in value.items():
            edge = (head, tail)
            edgeList.append(edge)
    return edgeList, g_adj

# return the subgraph's path
def getSubgraphPath(edgeList, g_adj):
    G = nx.DiGraph(edgeList)
    roots = list(v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)
    AllLeaves = list(set(nx.nodes(G)).difference(set(roots)))
    AllRoots = list(set(nx.nodes(G)).difference(set(leaves)))
    all_paths = []
    relationPaths = []
    if len(roots) ==0 :
        r = g_adj[edgeList[0][0]][edgeList[0][1]]#WQ
        # r = g_adj[edgeList[0][0]][edgeList[0][1]][0]#PQ
        if r.find('/') >=0 :
            r = r.strip().split('/')[-1]
        relationPaths.append([r])  
        return relationPaths
    for root in AllRoots:
        for leaf in AllLeaves:
            paths = nx.all_simple_paths(G, root, leaf)                    
            all_paths.extend(paths)    
    for path in all_paths:
        i = 0
        relationPath = []
        length = len(path)
        # print("length:", length)
        # print("path:", path)
        if i <= length - 2:
            r = g_adj[path[i]][path[i+1]] # Dataset WQ
            # r = g_adj[path[i]][path[i+1]][0] # Dataset PQ

            if r.find('/') >=0 :
                r = r.strip().split('/')[-1] 
            # print("r:", r)
            relationPath.append(r)
            i = i + 1
        relationPaths.append(relationPath)    
    return relationPaths

# calculate two relationPaths' similarity
def similarityPath(relationPaths1, relationPaths2):    
    len1 = len(relationPaths1)
    len2 = len(relationPaths2)
    # print("len1:", len1, relationPaths1)
    # print("len2:", len2, relationPaths2)
    count = 0
    visitIndex1 = []
    visitIndex2 = []
    for i, path1 in enumerate(relationPaths1):
        for j, path2 in enumerate(relationPaths2):
            if(path1 == path2) and (i not in visitIndex1) and (j not in visitIndex2):
                count +=1
                visitIndex1.append(i)
                visitIndex2.append(j)
                break
    similarity = count/(len1 + len2 - count)
    return similarity

if __name__=='__main__':
    input_dir = './dataset/WQ'
    mode = 'test' 
    structureSimi(input_dir, mode, k = 20)
