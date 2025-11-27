import sys
import random
import math

# --- Khai báo biến toàn cục (giống C++) ---
population = 100
gen = 500
seed = 30
city = 0
cost = []
bestfit = []
par = []

# --- PHẦN HỌC TĂNG CƯỜNG (RL-M) ---
RQ = []
ALPHA = 0.1  # Tốc độ học
GAMMA = 0.9  # Hệ số chiết khấu
N_MUTATION_LOOPS = 50  # Đây là 'N' trong Algorithm 2
N_CROSS_ITERATIONS = 20  # Đây là N_cross

# --- get fitness ---
def get_fitness(x):
    """Tính toán chi phí (fitness) của một tour."""
    fit = 0
    for i in range(city):
        # x[i] và x[(i + 1) % city] là các thành phố (1-based)
        fit += cost[x[i]][x[(i + 1) % city]]
    return fit

# --- get average best fitness in current generation ---
def getmax(x):
    """Tìm fitness tốt nhất (nhỏ nhất) trong thế hệ hiện tại và cộng vào bestfit."""
    global bestfit
    maxfit = float('inf')
    for i in range(1, population + 1):
        distance = get_fitness(par[i])
        if distance < maxfit:
            maxfit = distance
    bestfit[x] += maxfit

# --- check (đo độ tương đồng) ---
def check(u, v):
    """Kiểm tra độ tương đồng cạnh giữa hai cá thể u và v."""
    same = 0
    edges_u = set()
    
    # Lấy các cạnh từ cha u
    tour_u = par[u]
    for i in range(city):
        u1 = tour_u[i]
        v1 = tour_u[(i + 1) % city]
        # Thêm cạnh đã sắp xếp để xử lý (1, 2) và (2, 1) là như nhau
        edges_u.add(tuple(sorted((u1, v1))))

    # Đếm cạnh trùng với cha v
    tour_v = par[v]
    for i in range(city):
        u1 = tour_v[i]
        v1 = tour_v[(i + 1) % city]
        if tuple(sorted((u1, v1))) in edges_u:
            same += 1
            
    return same / city

# --- create_offspring (từ EAX) ---
def create_offspring(u, v, ABcycles):
    """Tạo con từ các AB-cycles (Bước 3-5 của EAX)."""
    
    # Bước 1: Lấy tất cả các cạnh từ parent A (par[u])
    edge = set()
    tour_u = par[u]
    for i in range(city):
        u1 = tour_u[i]
        v1 = tour_u[(i + 1) % city]
        edge.add(tuple(sorted((u1, v1))))
        
    # Bước 2: Flip các cạnh trong AB-cycles được chọn
    for cycle in ABcycles:
        if random.randint(0, 1) == 0:  # Ngẫu nhiên chọn cycle
            continue
        
        # C++ dùng cycle.size() - 1, vì cycle[0] == cycle.back()
        for j in range(len(cycle) - 1):
            u1 = cycle[j]
            v1 = cycle[j + 1]
            e = tuple(sorted((u1, v1)))
            
            if e in edge:
                edge.remove(e)
            else:
                edge.add(e)
                
    # Bước 3: Xây dựng đồ thị (adjacency list) từ các cạnh
    adj = {i: [] for i in range(1, city + 1)}
    for u1, v1 in edge:
        adj[u1].append(v1)
        adj[v1].append(u1)
        
    # Bước 4: Tìm các subtour
    subtours = []
    visited = set()
    
    for start_node in range(1, city + 1):
        if start_node in visited:
            continue
            
        tour = []
        curr = start_node
        visited.add(curr)
        tour.append(curr)
        
        # Duyệt theo cạnh để tạo thành đường đi (subtour)
        while True:
            found = False
            for next_node in adj[curr]:
                if next_node not in visited:
                    visited.add(next_node)
                    tour.append(next_node)
                    curr = next_node
                    found = True
                    break
            if not found:
                break
        
        subtours.append(tour)
        
    # Bước 5: Nối các subtour lại với nhau
    while len(subtours) > 1:
        best_i = -1
        best_j = -1
        pos_i = -1
        pos_j = -1
        min_cost = float('inf')

        # Tìm cặp subtour và vị trí tốt nhất để nối
        for i in range(len(subtours)):
            for j in range(i + 1, len(subtours)):
                tour1 = subtours[i]
                tour2 = subtours[j]
                
                for pi in range(len(tour1)):
                    for pj in range(len(tour2)):
                        a1 = tour1[pi]
                        a2 = tour1[(pi + 1) % len(tour1)]
                        b1 = tour2[pj]
                        b2 = tour2[(pj + 1) % len(tour2)]
                        
                        # Chi phí nối (2-opt move)
                        delta = cost[a1][b1] + cost[a2][b2] - cost[a1][a2] - cost[b1][b2]
                        
                        if delta < min_cost:
                            min_cost = delta
                            best_i = i
                            best_j = j
                            pos_i = pi
                            pos_j = pj
        
        # Nối 2 subtour lại
        merged = []
        tour_i = subtours[best_i]
        tour_j = subtours[best_j]
        
        # Thêm phần đầu của subtour i
        merged.extend(tour_i[0 : pos_i + 1])
        
        # Thêm subtour j (đã xoay)
        merged.extend(tour_j[pos_j + 1 :])
        merged.extend(tour_j[0 : pos_j + 1])
        
        # Thêm phần còn lại của subtour i
        merged.extend(tour_i[pos_i + 1 :])
        
        # Cập nhật danh sách subtours
        new_subtours = []
        for i in range(len(subtours)):
            if i != best_i and i != best_j:
                new_subtours.append(subtours[i])
        new_subtours.append(merged)
        subtours = new_subtours

    return subtours[0]


# --- EAX (Edge Assembly Crossover) ---
def EAX(u, v):
    """Thực hiện lai ghép EAX (Algorithm 1) - Bước 1 & 2."""
    edge_type = {}
    visited = {}
    Aedge = []
    Bedge = []
    
    tour_u = par[u]
    tour_v = par[v]

    # --- Lấy cạnh của cha A ---
    for i in range(city):
        x = tour_u[i]
        y = tour_u[(i + 1) % city]
        edge = tuple(sorted((x, y)))
        Aedge.append(edge)
        edge_type[edge] = 1
        
    # --- Lấy cạnh của cha B ---
    for i in range(city):
        x = tour_v[i]
        y = tour_v[(i + 1) % city]
        edge = tuple(sorted((x, y)))
        if edge in edge_type:  # bỏ trùng
            continue
        Bedge.append(edge)
        edge_type[edge] = 2
        
    # --- Gộp các cạnh ---
    ABedge = Aedge + Bedge
    
    # --- Tìm AB-cycle ---
    adj = {i: [] for i in range(1, city + 1)}
    for x, y in ABedge:
        adj[x].append(y)
        adj[y].append(x)
        visited[tuple(sorted((x,y)))] = False

    ABcycle = []
    
    for e in ABedge:
        edge = tuple(sorted(e))
        if visited[edge]:
            continue
            
        cycle = []
        start_node = e[0]
        curr_node = e[1]
        cycle.append(start_node)
        cycle.append(curr_node)
        visited[edge] = True
        curr_type = edge_type[edge]
        
        while True:
            # Chuyển loại cạnh (1 -> 2, 2 -> 1)
            curr_type = 3 - curr_type 
            found = False
            
            # Tìm cạnh tiếp theo cùng loại `curr_type`
            for next_node in adj[curr_node]:
                next_edge = tuple(sorted((curr_node, next_node)))
                
                # Bỏ qua nếu: không phải cạnh AB, đã thăm, hoặc sai loại
                if next_edge not in edge_type or visited[next_edge] or edge_type[next_edge] != curr_type:
                    continue

                visited[next_edge] = True
                cycle.append(next_node)
                curr_node = next_node
                found = True
                break
                
            if not found or curr_node == start_node:
                break
                
        # --- Kiểm tra chu trình hợp lệ ---
        if len(cycle) > 2 and cycle[0] == cycle[-1]:
            ABcycle.append(cycle)

    # --- Tạo con từ AB-cycle ---
    child_tour = create_offspring(u, v, ABcycle)
    return child_tour


# --- RL-M (Reinforcement Learning Mutation) ---
def RL_M(individual_X, idx):
    """Thực hiện đột biến RL-M (Algorithm 2)."""
    global RQ # Sửa đổi Q-table toàn cục
    
    len_X = get_fitness(individual_X)
    Y = []  # Tour Y
    len_Y = float('inf')
    
    count = 0  #
    
    while True: # Tương đương do-while
        # --- Bắt đầu 1 lần đột biến ---
        visited = set()
        new_tour = [] # Đây là 'Y'
        
        start_city = individual_X[0]
        new_tour.append(start_city) # Y[0] <- X[0]
        visited.add(start_city)
        
        curr = start_city # t <- Y[0]
        
        # Lấy ra các cạnh của tour X (individual_X)
        next_in_X = {}
        for i in range(city):
            next_in_X[individual_X[i]] = individual_X[(i + 1) % city]

        len_Y = 0  #

        # Vòng lặp for(k=1 to n-1)
        while len(new_tour) < city:
            best_opt = -1
            best_rq = -float('inf')
            
            # q <- chọn thành phố s trong J(t) theo max RQ(t,s)
            for s in range(1, city + 1):
                if s not in visited:  # s thuộc J(t)
                    if RQ[curr][s] > best_rq:
                        best_rq = RQ[curr][s]
                        best_opt = s

            # Đây là logic cốt lõi của bài báo
            nx_city_in_X = next_in_X[curr] # Đây là X[i+1]
            choose = -1

            # Nếu X[i+1] thuộc J(t) VÀ d(t, X[i+1]) < d(t, s)
            nx_unvisited = (nx_city_in_X not in visited)
            
            if nx_unvisited and (best_opt == -1 or cost[curr][nx_city_in_X] < cost[curr][best_opt]):
                choose = nx_city_in_X  # q <- X[i+1]
            else:
                choose = best_opt  # q <- s
            
            visited.add(choose)
            new_tour.append(choose)
            len_Y += cost[curr][choose] #

            # Cập nhật RQ
            max_rq_next = -float('inf')
            for z in range(1, city + 1):
                if z not in visited:  # z thuộc J(q)
                    if RQ[choose][z] > max_rq_next:
                        max_rq_next = RQ[choose][z]
            
            if max_rq_next == -float('inf'):
                max_rq_next = 0.0 # Xử lý bước cuối cùng

            ux = RQ[curr][choose]
            reward = 1.0 / cost[curr][choose]
            RQ[curr][choose] = (1.0 - ALPHA) * ux + ALPHA * (reward + GAMMA * max_rq_next)
            
            curr = choose # t <- q
        
        # Hoàn thành tour
        start_city = new_tour[0]
        len_Y += cost[curr][start_city]

        # Cập nhật RQ cho cạnh cuối cùng
        ux = RQ[curr][start_city]
        reward = 1.0 / cost[curr][start_city]
        RQ[curr][start_city] = (1.0 - ALPHA) * ux + ALPHA * reward
        
        # --- Kết thúc 1 lần đột biến ---
        
        count += 1  #
        Y = new_tour # Gán tour Y
        
        # Điều kiện dừng: "Until (Len Y < Len X or Count = N)"
        if not (len_Y >= len_X and count < N_MUTATION_LOOPS):
            break

    # "If Len Y < Len X then X <- Y otherwise retain X"
    if len_Y < len_X:
        return Y  # Trả về Y
    else:
        return individual_X  # Trả về X

# --- Hàm main ---
def main():
    global city, cost, bestfit, par, RQ
    
    # Chuyển hướng stdin/stdout
    try:
        sys.stdin = open("tsp_ga/cost101.txt", "r")
        sys.stdout = open("tsp_ga/best_avg_reinforcement101.txt", "w")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'cost.txt'")
        return

    # Đọc input
    city = int(sys.stdin.readline())
    
    # --- Khởi tạo các mảng toàn cục với kích thước (N+1) ---
    # (Để sử dụng 1-based indexing giống C++)
    cost = [[0.0] * (city + 1) for _ in range(city + 1)]
    RQ = [[0.0] * (city + 1) for _ in range(city + 1)]
    bestfit = [0.0] * (gen + 1)
    par = [None] * (population + 1) # par[0] sẽ không được dùng

    for i in range(1, city + 1):
        line = list(map(float, sys.stdin.readline().split()))
        for j in range(1, city + 1):
            cost[i][j] = line[j-1] # Đọc từ input 0-based vào cost 1-based

    # --- Vòng lặp các lần chạy (seed) ---
    for test in range(seed):
        # Khởi tạo Q-table
        for i in range(1, city + 1):
            for j in range(1, city + 1):
                if i != j:
                    RQ[i][j] = 1.0 / cost[i][j] if cost[i][j] > 0 else 0
        
        # Khởi tạo quần thể
        bg = list(range(1, city + 1))
        for i in range(1, population + 1):
            random.shuffle(bg)
            par[i] = list(bg) # Tạo bản sao
            
        getmax(1) # Ghi nhận thế hệ đầu tiên
        
        # --- Vòng lặp GA chính (Algorithm 3: RMGA) ---
        for i in range(2, gen + 1):
            
            # --- BƯỚC 1: H-EAX Crossover (Algorithm 1) ---
            for k in range(1, population + 1):
                # k là "family father" (A_i)
                par1_idx = k
                
                # Chọn "family mother" (A_j) ngẫu nhiên, j != i
                par2_idx = par1_idx
                while par2_idx == par1_idx:
                    par2_idx = random.randint(1, population)

                # Giữ lại tối ưu của bạn (kiểm tra tương đồng)
                if check(par1_idx, par2_idx) >= 0.5:
                    continue
                    
                # --- Vòng lặp N_cross ---
                best_child_from_n_cross = None
                best_child_fitness = float('inf')

                for n in range(N_CROSS_ITERATIONS):
                    # "lặp lại N_cross lần EAX(A_i, A_j)"
                    current_child = EAX(par1_idx, par2_idx)
                    current_fitness = get_fitness(current_child)
                    
                    if current_fitness < best_child_fitness:
                        best_child_fitness = current_fitness
                        best_child_from_n_cross = current_child
                
                # `best_child_from_n_cross` bây giờ là 'X'

                # Algorithm 1: "If X is better than A_i, then A_i <- X"
                if best_child_fitness < get_fitness(par[par1_idx]):
                    par[par1_idx] = best_child_from_n_cross
                

            # --- BƯỚC 2: ĐỘT BIẾN (RL-M) (Algorithm 2 & 3) ---
            for k in range(1, population + 1):
                par[k] = RL_M(par[k], k)
                
            print(f"Đã chạy tăng cường gen{i} seed {test + 1}/{seed}", file=sys.stderr)
                
            # --- BƯỚC 3: Ghi nhận bestfit ---
            getmax(i)
            # In ra stderr để theo dõi tiến độ (tùy chọn)
            # print(f"Seed {test+1}/{seed}, Gen {i}/{gen}", file=sys.stderr)
        print(f"Đã chạy xong seed {test + 1}/{seed}", file=sys.stderr)

    # In kết quả trung bình
    for i in range(1, gen + 1):
        print(bestfit[i] / seed)

# --- Điểm vào của chương trình ---
if __name__ == "__main__":
    main()