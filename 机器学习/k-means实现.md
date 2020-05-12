# 计算新的聚类中心.  
```
def computeCentroids(X, idx, K):
  # X是样本矩阵, idx标记当前样本属于哪个聚类点, K是待聚类点的个数.  
  m, n = X.shape
  centroids = np.zeros(K, n)   # 聚类点矩阵.  
  
  for cent in range(K):   
      centroids[cent] = np.mean(X[idx == cent], axis=0)
  
  return centroids.
```

# 初始化聚类中心.  
```
def kMeansInitCentroids(X, K):
  m, n = X.shape 
  centroids = np.zeros((K, n))
  
  # 随机选k个样本点作为初始聚类中心.
  randidx = np.random.permutation(X.shape[0])
  centroids = X[randidx[:K], :]
  
  return centroids 
```

# 找最近的聚类中心.  
```
def findClosestCentroids(X, centroids):
  K = centroids.shape[0]
  idx = np.zeros(X.shape[0], dtype=int)
  
  for i in range(X.shape[0]):   # m个样本数
      min = float('inf')  
      min_idx = -1
      for j in range(K):  
          tmp = np.sqrt(sum((X[i]-centroids[j]) ** 2))
          if tmp<min:
              min = tmp
              min_idx = j
      idx[i] = min_idx
  return idx 
```

# 主程序.  
```
centroids = kMeansInitCentroids(X, K)
for i in range(iterations):
    # 计算得出 K个类 的聚类中心.
    idx = findClosestCentroids(X, centroids)
    
    # 计算K个类 的均值，更新聚类中心. 
    centroids = computeCentroids(X, idx, K)
```
