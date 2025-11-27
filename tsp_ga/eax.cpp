#include<bits/stdc++.h>
#include<random>
using namespace std;

random_device rd;
mt19937 g(rd());
std::uniform_int_distribution<int> dist_pop(1, 100);
typedef pair<int, int> ii;
const double ALPHA = 0.1; // Tốc độ học [cite: 256]
const double GAMMA = 0.9; // Hệ số chiết khấu [cite: 256]

int city, cost[200][200];
vector<int> par[200], new_par[200];
double RQ[204][202];
vector<int> RM_L(int u){
    memset(RQ, 0, sizeof(RQ));
    for(int i = 1; i <= city; i++)
    for(int j = 1; j <= city; j++)
        if(i != j)
            RQ[i][j] = 1.0 / cost[i][j];
    
    int count = 0, maxcount = 10;
    while(count < maxcount){
        map<int, int> visited;
        vector<int> new_tour;
        new_tour.push_back(par[u][0]);
        int curr = new_tour[0];
        int next_city[200];
        visited[curr] = 1;
        for(int i = 0; i < par[u].size(); i++)
            next_city[par[u][i]] = par[u][(i + 1) % city];
        while(new_tour.size() < par[u].size()){
            int best_opt, choose;
            double best_rq(-1e9);
            for(int i = 1; i <= city; i++)
                if(best_rq < RQ[curr][i] and visited[i] == 0){
                    best_opt = i;
                    best_rq = RQ[curr][i];
                }
            int nx_city = next_city[curr];
            if(visited[nx_city] == 0){
                if(cost[curr][nx_city] < cost[curr][best_opt])
                    choose = nx_city;
                else 
                    choose = best_opt;
            }
            else 
                choose = best_opt;
            
            visited[choose] = 1;
            new_tour.push_back(choose);
            double max_rq(-1e9);
            for(int i = 1; i <= city; i++)
                if(visited[i] == 0 and RQ[choose][i] > max_rq)
                    max_rq = RQ[choose][i];
            double ux = RQ[curr][choose];
            RQ[curr][choose] = (1 - ALPHA) * ux + ALPHA * (1.0 / cost[curr][choose] + GAMMA * max_rq);
            curr = choose;
        }
        
        return new_tour;
        //if(get_fitness(next_tour) > get_fitness(par[u]))
            //return next_tour;
        count++;
    }
    return par[u];
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

    vector<int> bg;
    for(int i = 1; i <= city; i++)
        bg.push_back(i);

    shuffle(bg.begin(), bg.end(), g);
    par[1] = bg;
    shuffle(bg.begin(), bg.end(), g);
    par[2] = bg;
    for(auto i : par[1])
        cout << i << " ";
    par[1] = RM_L(1);
    cout << '\n';
    for(auto i : par[1])
        cout << i << " ";
}