# 快排法

```
class Solution:
    def partition(self, nums, left, right):
        pivot = nums[left] 
        l = left 
        r = right 

        while l < r :
            while l < r and nums[r] >= pivot:
                r -= 1  
            if l < r :
                nums[l] = nums[r]
            while l < r and nums[l] <= pivot:
                l += 1
            if l < r :
                nums[r] = nums[l] 
        nums[l] = pivot 
        return l                  

    def findKthLargest(self, nums, k) :
        left = 0 
        right = len(nums) - 1
        target = right - k + 1 

        while left <= right :
            mid = self.partition(nums, left, right)
            if mid == target:
                return nums[mid] 
            elif mid < target:
                left = mid+1 
            else:
                right = mid-1
```

# 堆排序  

```
import heapq 
# 默认最小堆. 

class Solution:
    def findKthLargest(self, nums, k) :
        heap = [] 
        
        for v in nums:
            if len(heap) < k:
                heappush(heap, v)
            else:
                if v > heap[0]:
                    heappop(heap)
                    heappush(heap, v)
        return heap[0]
```
