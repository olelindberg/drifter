import math

Lx        = 1000000
dx_target = 100


p  = math.ceil(math.log(Lx/dx_target,2))
dx = Lx/2**p 

print(f"Domain size         : {Lx}")
print(f"Target element size : {dx_target}")
print(f"Number of levels    : {p}")
print(f"Element size        : {dx}")

