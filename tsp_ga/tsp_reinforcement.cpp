#include<bits/stdc++.h>
#include<random>
using namespace std;

typedef pair<double, int> ii;
typedef pair<int, int> iii;
typedef vector<set<int>> pii;
int population = 100, gen = 500, seed = 30;
int city, cost[200][200], bestfit[505];
vector<int> par[200]; 

// --- PHẦN HỌC TĂNG CƯỜNG (RL-M) ---
double RQ[200][200]; // Bảng Q-Table (1-based)
const double ALPHA = 0.1; // Tốc độ học [cite: 256]
const double GAMMA = 0.9; // Hệ số chiết khấu [cite: 256]
// N_MUTATIONS là số lần lặp đột biến (Algorithm 2, dòng 161)
const int N_MUTATION_LOOPS = 50; // Đây là 'N' trong Algorithm 2
const int N_CROSS_ITERATIONS = 20; // Đây là N_cross [cite: 256]
// --- KẾT THÚC PHẦN RL-M ---

random_device rd;
mt19937 g(rd());
std::uniform_int_distribution<int> dist_pop(1, population);
std::uniform_real_distribution<double> rate_dist(0.0, 1.0);



double check(int u, int v){
    int same = 0;
    map<int, int> mp[200];
    
    for(int i = 0; i < par[u].size(); i++){
        int u1 = par[u][i];
        int v1 = par[u][(i + 1) % par[u].size()]; 
        mp[u1][v1] = 1;
        mp[v1][u1] = 1;
    }

    for(int i = 0; i < par[v].size(); i++){
        int u1 = par[v][i];
        int v1 = par[v][(i + 1) % par[v].size()]; 
        if(mp[u1][v1] == 1 || mp[v1][u1] == 1)
            same++;
    }
    return (double)same / city;
}

// get fitness
int get_fitness(vector<int> x){
    int fit(0);
    for(int i = 0; i < city; i++)
        fit += cost[x[i]][x[(i + 1) % city]]; 
    return fit;
}

// get average best fitness in current generation
void getmax(int x){
    int maxfit = 1e9;
    for(int i = 1; i <= population; i++){
        int distance = get_fitness(par[i]);
        if(distance < maxfit)
            maxfit = distance;
    }
    bestfit[x] += maxfit;
}

vector<int> create_offspring(int u, int v, vector<vector<int>> &ABcycles){
    
    // Bước 1: Lấy tất cả các cạnh từ parent A
    set<pair<int,int>> edge;
    for(int i = 0; i < par[u].size(); i++){
        int u1 = par[u][i];
        int v1 = par[u][(i + 1) % par[u].size()];
        if(u1 > v1) swap(u1, v1);
        edge.insert({u1, v1});
    }
    
    // Bước 2: Flip các cạnh trong AB-cycles được chọn
    for(int i = 0; i < ABcycles.size(); i++){
        if(dist_pop(g) % 2 == 0) continue; // Ngẫu nhiên chọn cycle
        
        for(int j = 0; j < ABcycles[i].size() - 1; j++){
            int u1 = ABcycles[i][j];
            int v1 = ABcycles[i][j + 1];
            if(u1 > v1) swap(u1, v1);
            
            if(edge.count({u1, v1}))
                edge.erase({u1, v1});
            else
                edge.insert({u1, v1});
        }
    }

    // Bước 3: Xây dựng đồ thị từ các cạnh
    map<int, vector<int>> adj;
    map<int, int> degree;
    
    for(auto &e : edge){
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        degree[e.first]++;
        degree[e.second]++;
    }

    // Bước 4: Tìm các subtour
    vector<vector<int>> subtours;
    set<int> visited;

    for (int start = 1; start <= city; start++) {
        if (visited.count(start)) continue;
        
        vector<int> tour;
        int curr = start;
        visited.insert(curr);
        tour.push_back(curr);
        
        // Duyệt theo cạnh để tạo thành đường đi
        while(true) {
            bool found = false;
            for(int next : adj[curr]) {
                if(!visited.count(next)) {
                    visited.insert(next);
                    tour.push_back(next);
                    curr = next;
                    found = true;
                    break;
                }
            }
            if(!found) break;
        }
        
        subtours.push_back(tour);
    }

    // Bước 5: Nối các subtour lại với nhau
    while (subtours.size() > 1) {
        int best_i = -1, best_j = -1;
        int pos_i = -1, pos_j = -1; // Vị trí để chèn
        int min_cost = 1e9;

        // Tìm cặp subtour và vị trí tốt nhất để nối
        for (int i = 0; i < subtours.size(); i++) {
            for (int j = i + 1; j < subtours.size(); j++) {
                // Thử tất cả các cách nối giữa 2 subtour
                for (int pi = 0; pi < subtours[i].size(); pi++) {
                    for (int pj = 0; pj < subtours[j].size(); pj++) {
                        int a1 = subtours[i][pi];
                        int a2 = subtours[i][(pi + 1) % subtours[i].size()];
                        int b1 = subtours[j][pj];
                        int b2 = subtours[j][(pj + 1) % subtours[j].size()];
                        
                        // Chi phí nối: bỏ 2 cạnh (a1,a2) và (b1,b2), 
                        // thêm 2 cạnh (a1,b1) và (a2,b2)
                        int delta = cost[a1][b1] + cost[a2][b2] 
                                  - cost[a1][a2] - cost[b1][b2];
                        
                        if (delta < min_cost) {
                            min_cost = delta;
                            best_i = i;
                            best_j = j;
                            pos_i = pi;
                            pos_j = pj;
                        }
                    }
                }
            }
        }

        // Nối 2 subtour lại
        vector<int> merged;
        
        // Thêm phần đầu của subtour i (từ 0 đến pos_i)
        for(int k = 0; k <= pos_i; k++) {
            merged.push_back(subtours[best_i][k]);
        }
        
        // Thêm subtour j (từ pos_j+1 đến hết, rồi từ đầu đến pos_j)
        for(int k = pos_j + 1; k < subtours[best_j].size(); k++) {
            merged.push_back(subtours[best_j][k]);
        }
        for(int k = 0; k <= pos_j; k++) {
            merged.push_back(subtours[best_j][k]);
        }
        
        // Thêm phần còn lại của subtour i (từ pos_i+1 đến hết)
        for(int k = pos_i + 1; k < subtours[best_i].size(); k++) {
            merged.push_back(subtours[best_i][k]);
        }

        // Cập nhật danh sách subtours
        vector<vector<int>> new_subtours;
        for (int i = 0; i < subtours.size(); i++) {
            if (i != best_i && i != best_j) {
                new_subtours.push_back(subtours[i]);
            }
        }
        new_subtours.push_back(merged);
        subtours = new_subtours;
    }

    // Trả về tour hoàn chỉnh
    return subtours[0];
}
// EAX
vector<int> EAX(int u, int v) {
    map<pair<int,int>, int> type, visited;
    vector<pair<int,int>> ABedge, Aedge, Bedge;

    // --- Lấy cạnh của cha A ---
    for (int i = 0; i < city; i++) {
        int x = par[u][i], y = par[u][(i + 1) % city];
        if (x > y) swap(x, y);
        Aedge.push_back({x, y});
        type[{x, y}] = 1;
    }

    // --- Lấy cạnh của cha B ---
    for (int i = 0; i < city; i++) {
        int x = par[v][i], y = par[v][(i + 1) % city];
        if (x > y) swap(x, y);
        if (type[{x, y}] == 1) continue; // bỏ trùng
        Bedge.push_back({x, y});
        type[{x, y}] = 2;
    }

    // --- Gộp các cạnh ---
    ABedge.insert(ABedge.end(), Aedge.begin(), Aedge.end());
    ABedge.insert(ABedge.end(), Bedge.begin(), Bedge.end());

    // --- Tìm AB-cycle ---
    vector<vector<int>> ABcycle;
    for (auto e : ABedge) {
        pair<int,int> edge = e;
        if (edge.first > edge.second) swap(edge.first, edge.second);
        if (visited[edge]) continue;

        vector<int> cycle;
        int start = e.first;
        int curr = e.second;
        cycle.push_back(start);
        cycle.push_back(curr);
        visited[edge] = 1;
        int curr_type = type[e];

        while (true) {
            curr_type = (curr_type == 1) ? 2 : 1;
            bool found = false;
            pair<int,int> next_edge;
            int next_node = -1;

            for (auto ne : ABedge) {
                pair<int,int> temp = ne;
                if (temp.first > temp.second) swap(temp.first, temp.second);
                if (visited[temp] || type[ne] != curr_type) continue;

                if (ne.first == curr) { next_edge = ne; next_node = ne.second; found = true; break; }
                if (ne.second == curr) { next_edge = ne; next_node = ne.first; found = true; break; }
            }

            if (!found) break;

            pair<int,int> temp = next_edge;
            if (temp.first > temp.second) swap(temp.first, temp.second);
            visited[temp] = 1;
            cycle.push_back(next_node);
            if (next_node == start) break;
            curr = next_node;
        }

        // --- Kiểm tra chu trình hợp lệ ---
        if (cycle.size() > 2 && cycle.front() == cycle.back())
            ABcycle.push_back(cycle);
    }

    // --- Tạo con từ AB-cycle ---
    vector<int> child_tour = create_offspring(u, v, ABcycle);

    return child_tour;
}


vector<int> RL_M(vector<int> individual_X) {
    int len_X = get_fitness(individual_X);
    vector<int> Y; // Tour Y [cite: 137]
    int len_Y;
    
    int count = 0; // [cite: 140]

  
    do {
        // --- Bắt đầu 1 lần đột biến [cite: 147] ---
        map<int, int> visited;
        vector<int> new_tour; // Đây là 'Y'
        new_tour.push_back(individual_X[0]); // Y[0] <- X[0] [cite: 143]
        
        int curr = new_tour[0]; // t <- Y[0] [cite: 143]
        visited[curr] = 1;

        // Lấy ra các cạnh của tour X (individual_X)
        map<int, int> next_in_X;
        for(int i = 0; i < individual_X.size(); i++)
            next_in_X[individual_X[i]] = individual_X[(i + 1) % city];

        len_Y = 0; // [cite: 146]

       // Vòng lặp for(k=1 to n-1) [cite: 147]
        while(new_tour.size() < city) { 
            int best_opt = -1;
            double best_rq = -1e9;
            
            // q <- chọn thành phố s trong J(t) theo max RQ(t,s) [cite: 149]
            for(int s = 1; s <= city; s++) {
                if (visited.count(s) == 0) { // s thuộc J(t)
                    if(RQ[curr][s] > best_rq) {
                        best_rq = RQ[curr][s];
                        best_opt = s;
                    }
                }
            }

            // Đây là logic cốt lõi của bài báo [cite: 151]
            int nx_city_in_X = next_in_X[curr]; // Đây là X[i+1]
            int choose;

            // Nếu X[i+1] thuộc J(t) VÀ d(t, X[i+1]) < d(t, s) [cite: 151]
            if (visited.count(nx_city_in_X) == 0 && cost[curr][nx_city_in_X] < cost[curr][best_opt]) {
                choose = nx_city_in_X; // q <- X[i+1]
            } else {
                choose = best_opt; // q <- s
            }
            
            visited[choose] = 1;
            new_tour.push_back(choose);
            len_Y += cost[curr][choose]; // [cite: 152]

            // Cập nhật RQ [cite: 159]
            double max_rq_next = -1e9;
            for(int z = 1; z <= city; z++) {
                if(visited.count(z) == 0) { // z thuộc J(q)
                    if (RQ[choose][z] > max_rq_next)
                        max_rq_next = RQ[choose][z];
                }
            }
            if (max_rq_next == -1e9) max_rq_next = 0; // Xử lý bước cuối cùng

            double ux = RQ[curr][choose];
            RQ[curr][choose] = (1 - ALPHA) * ux + ALPHA * (1.0 / cost[curr][choose] + GAMMA * max_rq_next);
            
            curr = choose;// t <- q [cite: 159]
        }
        
        // Hoàn thành tour [cite: 158]
        int start_city = new_tour[0];
        len_Y += cost[curr][start_city]; 

        // Cập nhật RQ cho cạnh cuối cùng [cite: 159]
        double ux = RQ[curr][start_city];
        RQ[curr][start_city] = (1 - ALPHA) * ux + ALPHA * (1.0 / cost[curr][start_city]); 
        
        // --- Kết thúc 1 lần đột biến [cite: 160] ---
        
        count++; // [cite: 160]
        Y = new_tour; // Gán tour Y
        
        // Điều kiện dừng: "Until (Len Y < Len X or Count = N)" 
    } while (len_Y >= len_X && count < N_MUTATION_LOOPS);

    // "If Len Y < Len X then X <- Y otherwise retain X" [cite: 162]
    if (len_Y < len_X) {
        return Y; // Trả về Y
    } else {
        return individual_X; // Trả về X
    }
}
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    freopen("cost.txt", "r", stdin);
    freopen("best_avg_reinforcement.txt", "w", stdout); 
    
    cin >> city;
    for(int i = 1; i <= city; i++)
    for(int j = 1; j <= city; j++)
        cin >> cost[i][j];
    
    for(int test = 1; test <= seed; test++){
        for(int i = 1; i <= city; i++)
        for(int j = 1; j <= city; j++)
            if(i != j)
                RQ[i][j] = 1.0 / cost[i][j];
        vector<int> bg;
        for(int i = 1; i <= city; i++)
            bg.push_back(i);

        for(int i = 1; i <= population; i++){
            shuffle(bg.begin(), bg.end(), g);
            par[i] = bg;
        }
        getmax(1); 
        // --- Vòng lặp GA chính (Algorithm 3: RMGA) ---
        for(int i = 2; i <= gen; i++){
            
            // --- BƯỚC 1: H-EAX Crossover (Algorithm 1) ---

            for (int k = 1; k <= population; k++) {
                // k là "family father" (A_i) [cite: 112, 121]
                int par1_idx = k;
                
            // Chọn "family mother" (A_j) ngẫu nhiên, j != i [cite: 121]
                int par2_idx;
                do {
                    par2_idx = dist_pop(g); 
                } while (par2_idx == par1_idx);

                // (Giữ lại tối ưu của bạn)
                if(check(par1_idx, par2_idx) >= 0.5) continue; 

                // --- Vòng lặp N_cross ---
                vector<int> best_child_from_n_cross;
                int best_child_fitness = 1e9;

                for (int n = 0; n < N_CROSS_ITERATIONS; n++) { 
                    // "lặp lại N_cross lần EAX(A_i, A_j)" 
                    vector<int> current_child = EAX(par1_idx, par2_idx); 
                    int current_fitness = get_fitness(current_child);

                    if (current_fitness < best_child_fitness) {
                        best_child_fitness = current_fitness;
                        best_child_from_n_cross = current_child;
                    }
                }
                // `best_child_from_n_cross` bây giờ là 'X' 

                // Algorithm 1: "If X is better than A_i, then A_i <- X" 
                // Đây là mấu chốt của "quần thể hỗn hợp" [cite: 113]
                if (get_fitness(best_child_from_n_cross) < get_fitness(par[par1_idx])) {
                    par[par1_idx] = best_child_from_n_cross; 
                }
            }
            
            // --- BƯỚC 2: ĐỘT BIẾN (RL-M) (Algorithm 2 & 3) ---
            for(int k = 1; k <= population; k++)
                par[k] = RL_M(par[k]);  
            
            // --- BƯỚC 3: Ghi nhận bestfit ---
            getmax(i);
        }
    }
    // In kết quả
    for(int i = 1; i <= gen; i++){
        cout << bestfit[i] / seed << "\n";
    }
}