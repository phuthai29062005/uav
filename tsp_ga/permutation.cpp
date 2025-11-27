#include<bits/stdc++.h>
#include<random>
using namespace std;


typedef pair<double, int> ii;
random_device rd;
mt19937 g(rd());

int city, cost[200][200];
double fitness[200], bestfit[505];
vector<int> par[200];

int population = 100, gen = 500, seed = 30;
double pc = 0.9, pm = 0.1;
std::uniform_int_distribution<int> dist(1, population);
std::uniform_real_distribution<double> rate_dist(0.0, 1.0);

vector<int> create_gen(int u, int v){
    vector<int> new_gen;
    int dis[100], val[100];
    int gen[100];
    memset(gen, 0, sizeof(gen));
    for(int i = 0; i < par[u].size(); i++){
        dis[i + 1] = par[u][i];
        val[par[u][i]] = i + 1;
    }
    int x = 1;
    while(1){
        if(gen[x] != 0)
            break;
        gen[x] = dis[x];
        x = val[par[v][x - 1]];
    }
    for(int i = 0; i < par[u].size(); i++){

        if(gen[i + 1] == 0)
            gen[i + 1] = par[v][i];
        new_gen.push_back(gen[i + 1]); 
    }
    return new_gen;
    
}

vector<int> create_gen_pmx(int u, int v){

    vector<int> new_gen;
    int dis[100][3];
    int gen[100];
    bool check[100];
    memset(gen, -1, sizeof(gen));
    memset(dis, -1, sizeof(dis));
    memset(check, false, sizeof(check));
    std::uniform_int_distribution<int> dist(0, city - 1);
    int x = dist(g);
    int y = dist(g);
    while(y < x)
        y = dist(g);
    
    for(int i = x; i <= y; i++){
        gen[i] = par[u][i];
        dis[par[u][i]][0] = par[v][i];
        dis[par[v][i]][1] = par[u][i]; 
        check[par[u][i]] = true;
    }


    for(int i = 0; i < city; i++){
        if(i >= x and i <= y)
            continue;
        int k = par[v][i];
        int count = 0;
        
       
        while(check[k] == true && count < city){
            k = dis[k][0]; 
            count++;
        }
        
        gen[i] = k;
    }

    for(int i = 0; i < city; i++)
        new_gen.push_back(gen[i]);
    
    return new_gen;

}

vector<int> create_gen_ox(int u, int v){
    vector<int> new_gen;
    int gen[200];
    bool check[200];
    memset(gen, -1, sizeof(gen));
    memset(check, false, sizeof(check));
    std::uniform_int_distribution<int> dist(0, city - 1);
    int x = dist(g);
    int y = dist(g);
    while(y < x)
        y = dist(g);

    for(int i = x; i <= y; i++){
        gen[i] = par[u][i];
        check[par[u][i]] = true;
    }
    deque<int> pq;
    
    for(int i = y + 1; i < city; i++)
         pq.push_back(par[v][i]);
    for(int i = 0; i <= y; i++)
        pq.push_back(par[v][i]);
    for(int i = y + 1; i < city; i++){
        while(check[pq.front()] == true)
            pq.pop_front();
        gen[i] = pq.front();
        pq.pop_front();
    }
    for(int i = 0; i < x; i++){
        while(check[pq.front()] == true)
            pq.pop_front();
        gen[i] = pq.front();
        pq.pop_front();
    }
    for(int i = 0; i < city; i++)
        new_gen.push_back(gen[i]);

    return new_gen;

}

void cswap(vector<int>& child){
    std::uniform_int_distribution<int> dist(0, city - 1);
    int u = dist(g);
    int v = dist(g);
    while(v == u)
        v = dist(g);
    swap(child[u], child[v]);
}

void getmax(int x){
    int maxfit = 1e9;
    for(int i = 1; i <= population; i++){
        int distance = 0;
        for(int u = 1; u < par[i].size(); u++)
            distance += cost[par[i][u]][par[i][u - 1]];
            distance += cost[par[i][0]][par[i][par[i].size() - 1]];
            maxfit = min(maxfit, distance);
    }
    bestfit[x] += maxfit;
    //cout << x << '\n';
}
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    freopen("cost101.txt", "r", stdin);
    freopen("best_avg_GAox101.txt", "w", stdout);
    cin >> city;
    for(int i = 1; i <= city; i++)
    for(int j = 1; j <= city; j++)
        cin >> cost[i][j];

    for(int test = 1; test <= seed; test++){
        // tao gen 1
        vector<int> bg;
        for(int i = 1; i <= city; i++)
            bg.push_back(i);
        
        for(int i = 1; i <= population; i++){
            shuffle(bg.begin(), bg.end(), g);
            par[i] = bg;
        }
        getmax(1);
        // tao cac gen tiep theo cx

        for(int i = 2; i <= gen; i++){

            // tính fitness của cha
            priority_queue<ii> pq;
            memset(fitness, 0, sizeof(fitness));
            for(int u = 1; u <= population; u++){
                for(int i = 1; i < par[u].size(); i++)
                    fitness[u] += cost[par[u][i]][par[u][i - 1]];
                fitness[u] += cost[par[u][0]][par[u][par[u].size() - 1]];
                fitness[u] = 1.0/fitness[u];
                pq.push({fitness[u], u});
            }
            vector<int> child[200];
            int cnt = 0;
            for(int j = 1; j <= population * pc / 2; j++){
                int u, v;

                // cha1 
                int u1 = dist(g), u2 = dist(g), u3 = dist(g);
                double um = max({fitness[u1], fitness[u2], fitness[u3]});
                if(fitness[u1] == um) u = u1;
                if(fitness[u2] == um) u = u2;
                if(fitness[u3] == um) u = u3;
                // cha2
                int v1 = dist(g), v2 = dist(g), v3 = dist(g);
                double vm = max({fitness[v1], fitness[v2], fitness[v3]});
                if(fitness[v1] == vm) v = v1;
                if(fitness[v2] == vm) v = v2;
                if(fitness[v3] == vm) v = v3;
                cnt++;
                child[cnt] = create_gen_ox(u, v);
                cnt++;
                child[cnt] = create_gen_ox(v, u);
            }
            while(cnt < population){
                cnt++;
                child[cnt] = par[pq.top().second];
                pq.pop();
            }

            // đột biến
            for(int k = 1; k <= population; k++){
                
                if(rate_dist(g) < pm)
                    cswap(child[k]);
            }
            for(int k = 1; k <= population; k++)
                par[k] = child[k];
            getmax(i);
        }
    }
    for(int i = 1; i <= gen; i++)
        cout << bestfit[i] / 30 << '\n';
    //cout << "Chạy thành công" << '\n';
}