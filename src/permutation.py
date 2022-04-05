def return_permutations():
    permuted_arr_list = []
    def permute(arr, l, r):
        if(l==r):
            permuted_arr_list.append(arr)
        else:
            for i in range(l, r+1):
                temp = arr[l]
                arr[l] = arr[i]
                arr[i] = temp
                
                permute(arr, l+1, r)
                
                temp2 = arr[l]
                arr[l] = arr[i]
                arr[i] = temp2
    
    arra = [0, 1, 2, 3, 4, 5, 6, 7]
    permute(arra, 0, 7)
    return permuted_arr_list

