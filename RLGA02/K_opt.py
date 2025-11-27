# loại bỏ import không dùng
# from ast import Return   <- bỏ
from read_data import cost_matrix
Kmax = 5

def get_fitness(tour):
    total_cost = 0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        total_cost += cost_matrix[u][v]
    return total_cost

def get_neighbor(p_2k, tour):
    neighbor = set()
    for i in range(len(tour)):
        u = tour[i]
        if u != p_2k:
            continue
        if i > 0:
            neighbor.add(tour[i - 1])
        if i < len(tour) - 1:
            neighbor.add(tour[i + 1])
    return neighbor

def the_constraint_C_I(tour, P):
    """
    tour: current tour (list of cities)
    P: current sequence p (list) extended by candidate (i.e., p ∪ {p_2k})
    Kiểm tra nếu ta remove/add edges như trong k-opt thì đồ thị vẫn là 1 chu trình Hamilton.
    """
    city_count = len(tour)
    # build adjacency from original tour
    adj = {city: [] for city in tour}
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        adj[u].append(v)
        adj[v].append(u)

    # P là sequence p (thứ tự p1, p2, ..., p_2k)
    # edges to remove: (p_{2j-1}, p_{2j})
    edges_to_remove = []
    for i in range(1, len(P) // 2 + 1):
        a = P[2 * i - 2]
        b = P[2 * i - 1]
        edges_to_remove.append((a, b))

    # edges to add: (p_{2j}, p_{2j+1}) for j=1..k-1, plus (p_2k, p_1)
    edges_to_add = []
    for j in range(1, len(P) // 2):
        a = P[2 * j - 1]
        b = P[2 * j]
        edges_to_add.append((a, b))
    # close the cycle
    edges_to_add.append((P[-1], P[0]))

    # apply removals/additions on adj (cẩn thận với thứ tự)
    try:
        for u, v in edges_to_remove:
            adj[u].remove(v)
            adj[v].remove(u)
        for u, v in edges_to_add:
            adj[u].append(v)
            adj[v].append(u)
    except ValueError:
        # cố gắng remove cạnh không tồn tại => move không hợp lệ
        return False

    # BFS/DFS để đếm số đỉnh thuộc thành phần liên thông chứa start
    start = tour[0]
    visited = {start}
    queue = [start]
    cnt = 0
    while queue:
        cur = queue.pop(0)
        cnt += 1
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return cnt == city_count

def apply_k_opt_move(tour, curr_tour):
    """
    Xây lại tour sau khi loại/ thêm các cạnh theo curr_tour.
    Nếu không thể tạo tour hợp lệ -> trả về None.
    """
    p1 = curr_tour[0]
    p2k = curr_tour[-1]

    # build adjacency
    adj = {city: [] for city in tour}
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        adj[u].append(v)
        adj[v].append(u)

    # remove edges (p_{2j-1}, p_{2j})
    for i in range(1, len(curr_tour) // 2 + 1):
        u = curr_tour[2 * i - 2]
        v = curr_tour[2 * i - 1]
        try:
            adj[u].remove(v)
            adj[v].remove(u)
        except ValueError:
            return None

    # add edges (p_{2j}, p_{2j+1}) for j=1..k-1
    for i in range(1, len(curr_tour) // 2):
        u = curr_tour[2 * i - 1]
        v = curr_tour[2 * i]
        adj[u].append(v)
        adj[v].append(u)

    # finally add (p_2k, p1)
    adj[p2k].append(p1)
    adj[p1].append(p2k)

    # now reconstruct the cycle deterministically:
    start = tour[0]
    new_tour = []
    cur = start
    prev = None
    for _ in range(len(tour)):
        new_tour.append(cur)
        # choose next neighbor that is not prev
        nxt = None
        for nb in adj[cur]:
            if nb != prev:
                nxt = nb
                break
        if nxt is None:
            return None
        prev, cur = cur, nxt

    # verify closed cycle (last connects to first)
    if new_tour[0] not in adj[new_tour[-1]]:
        return None

    # verify all cities present exactly once
    if len(set(new_tour)) != len(tour):
        return None

    return new_tour

def the_constraint_C_II(tour, curr_tour):
    k = (len(curr_tour) - 1) // 2
    total_removed = 0
    total_added = 0
    for j in range(1, k + 1):
        u_rem = curr_tour[2 * j - 2]
        v_rem = curr_tour[2 * j - 1]
        total_removed += cost_matrix[u_rem][v_rem]

    for j in range(1, k):
        u_add = curr_tour[2 * j - 1]
        v_add = curr_tour[2 * j]
        total_added += cost_matrix[u_add][v_add]

    # plus final added edge (p_2k, p1)
    total_added += cost_matrix[curr_tour[-1]][curr_tour[0]]

    return (total_removed > total_added)

def K_opt(tour, city, p, k, Nnear):
    """
    tour: current tour
    city: starting city p1 (root)
    p: current sequence [p1, p2, ..., p_{2k-1}]
    k: current depth (number of pairs already chosen)
    Nnear: neighbor/candidate list per city, e.g. Nnear[city] -> list
    """

    p_2k_1 = p[-1]
    neighbor = get_neighbor(p_2k_1, tour)

    for P_2K in neighbor:
        # check C-I for k >= 2 when we would close cycle later
        if k >= 2 and not the_constraint_C_I(tour, p + [P_2K]):
            continue

        curr_tour = p + [P_2K]

        if k >= 2:
            # compute sum removed / sum added following LKH formula
            sum_removed = 0
            sum_added = 0

            for j in range(1, k + 1):
                u = curr_tour[2 * j - 2]
                v = curr_tour[2 * j - 1]
                sum_removed += cost_matrix[u][v]

            for j in range(1, k):
                u = curr_tour[2 * j - 1]
                v = curr_tour[2 * j]
                sum_added += cost_matrix[u][v]

            # plus final candidate edge (p_2k, city)
            sum_added += cost_matrix[P_2K][city]

            if sum_added < sum_removed:
                x_out = apply_k_opt_move(tour, curr_tour)
                if x_out is not None and get_fitness(x_out) < get_fitness(tour):
                    return (x_out, curr_tour)

        if k == Kmax:
            return (tour, None)

        # expand: choose next p_{2k+1} from candidate list of P_2K
        # use top candidates Nnear[P_2K]
        candidates = Nnear.get(P_2K, []) if isinstance(Nnear, dict) else Nnear[P_2K]
        for j in range(min(Kmax, len(candidates))):
            P_2K_1 = candidates[j]
            if P_2K_1 in curr_tour:
                continue
            if not the_constraint_C_II(tour, curr_tour + [P_2K_1]):
                continue

            x_out, res_tour = K_opt(tour, city, curr_tour + [P_2K_1], k + 1, Nnear)
            if x_out is not None and get_fitness(x_out) < get_fitness(tour):
                return (x_out, res_tour)

    return (tour, None)
