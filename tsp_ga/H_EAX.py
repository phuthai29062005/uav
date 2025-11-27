



def EAX(par1, par2):
    
    ## Step 1: Tạo map cạnh và tập cạnh AB
    mp = dict()
    ABedge = set()
    for i in range(len(par1)):
        u = par1[i]
        v = par1[(i + 1) % len(par1)]
        edge = tuple(sorted((u, v))) # Chuẩn hóa cạnh
        mp[{edge}] = 1
        ABedge.add(edge)

    for i in range(len(par2)):
        u = par2[i]
        v = par2[(i + 1) % len(par2)]
        edge = tuple(sorted((u, v))) # Chuẩn hóa cạnh
        mp[{edge}] = 2
        ABedge.add(edge)

    ## Step 2: Tạo tập E-set
    