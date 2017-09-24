f(used, map) =
 check_used_stop + check_map_stop + change_data + n^2 * find + f(used-2, map-2) + f(used, map-2)

f(0, map) = check_used_stop

f(used, 2*(used-1)) = check_used_stop + check_map_stop

? f(n, n)


f(n, n) = 
    cu + cm + cd + n^2 * f
    f(n-2, n-2) 
        cu + cm + cd + n^2 * f
        f(n-4, n-4)
        f(n-2, n-4)
    f(n, n-2)
        cu + cm + cd + n^2 * f
        f(n-2, n-4)
        f(n, n-6)
