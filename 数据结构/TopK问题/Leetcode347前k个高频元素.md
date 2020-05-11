# 堆实现  

```
import heapq

class Solution:
    def topKFrequent(self, nums, k) :
        seek_dict = {} 
        heap = [] 

        for v in nums:
            if v in seek_dict:
                seek_dict[v] += 1
            else:
                seek_dict[v] = 1 
        
        for v in seek_dict:
            if len(heap) < k :
                heappush(heap, (seek_dict[v], v))
            else:
                if seek_dict[v] > heap[0][0]:
                    heappop(heap)
                    heappush(heap, (seek_dict[v], v))
        
        re = []
        for v in heap:
            re.append(v[1])
        return re 

```

# 建立哈希, 解决大样本的TOP k 高频问题.  
```
class Solution:
    def topKFrequent(self, nums, k) :
        seek_dict = {} 
        heap = [] 
        nums_len = len(nums)

        for v in nums:
            if v in seek_dict:
                seek_dict[v] += 1
            else:
                seek_dict[v] = 1 
        
        hash_list = [[] for _ in range(nums_len+1)]
        # hash最多_ 需要 数组长度. 

        for v in seek_dict:
            hash_list[seek_dict[v]].append(v) 
        
        count = 0 
        re = [] 
        for i in range(nums_len, 0, -1):
            if len(hash_list[i]) != 0 :
                for v in hash_list[i]:
                    re.append(v)
                    count += 1
                    if count == k:
                        return re 
```
