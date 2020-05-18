[Leetcode23](https://leetcode-cn.com/explore/interview/card/bytedance/244/linked-list-and-tree/1025/)

本来是逐个合并, 用了分治合并，其实本质就是归并排序.  

```
class Solution:
    def merge_two(self, head1, head2):
        re_head = ListNode(-1)
        cur = re_head
        
        while head1 and head2:
            if head1.val <= head2.val:
                cur.next = head1
                head1 = head1.next
            else:
                cur.next = head2
                head2 = head2.next
            cur = cur.next
        if head1:
            cur.next = head1
        if head2:
            cur.next = head2
        
        return re_head.next 
            
        
    def mergeKLists(self, lists) -> ListNode:
        if not lists:
            return [] 
        
        left = 0 
        right = len(lists)-1
        
        def dfs(left, right):
            if left+1 == right:
                return self.merge_two(lists[left], lists[right])
            if left == right :
                return lists[left]
            
            mid = (left + right) // 2 
            list_left = dfs(left, mid)
            list_right = dfs(mid+1, right)
            
            return self.merge_two(list_left, list_right)
        
        re = dfs(left, right)
        return re 
```
