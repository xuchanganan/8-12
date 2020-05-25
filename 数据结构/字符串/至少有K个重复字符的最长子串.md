[Leetcode 395.至少有K个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)  

```
class Solution:
    def longestSubstring(self, s: str, k) :
        s_len = len(s)
        
        if s_len < k:
            return 0
        s = list(s) 
        re_max = 0 

        for i in range(s_len):
            sa_dict = {}
            un_dict = {}
            cur_len = 0 

            for j in range(i, s_len):
                if s[j] not in sa_dict:
                    if s[j] not in un_dict:
                        un_dict[s[j]] = 1 
                    else:
                        un_dict[s[j]] += 1
                    if un_dict[s[j]] >= k:
                        sa_dict[s[j]] = un_dict[s[j]] 
                        del un_dict[s[j]]
                else:
                    sa_dict[s[j]] += 1 
                
                cur_len += 1
                if len(un_dict) == 0:
                    re_max = max(cur_len, re_max) 
        
        return re_max 
               
```
