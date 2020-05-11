# 双指针法_区间收缩.  
```
class Solution:
    def findClosestElements(self, arr, k, x) :
        left = 0 
        right = len(arr) - 1 
        del_num = len(arr) - k  

        while del_num :
            if abs(x - arr[left]) <= abs(x - arr[right]):
                right -= 1
            else:
                left += 1 
            del_num -= 1
        return arr[left: right+1] 
```

# 二分搜索
```
# 基于排除法的二分搜索.
    def findClosestElements(self, arr, k, x) :
        left = 0 
        right = len(arr) - k 

        while left < right:
            mid = (left + right) // 2 
            # 注意这里, 必须保证 mid    x     mid+k 这种情况, 所以不能用abs.
            # 反例: [1,1,2,2,2,2,2,3,3]
            if x - arr[mid] <= arr[mid+k] - x:
                right = mid 
            else:
                left = mid + 1

        return arr[left:left+k]
```
