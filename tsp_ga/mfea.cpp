#include<bits/stdc++.h>
#include<random>
using namespace std;

typedef pair<double, int> ii;
typedef pair<int, int> iii;
random_device rd;
mt19937 g(rd());


iii bestfit[505];
int city51, city101, cost51[150][150], cost101[150][150], type[550], newtype[550], parchil[550];
double fitness[550];
vector<int> par[550], child[550];
int population = 100, gen = 500, seed = 30;
double pc = 0.9, pm = 0.1, rmp = 0.3;
std::uniform_int_distribution<int> dist(1, population);
std::uniform_real_distribution<double> rate_dist(0.0, 1.0);

vector<int> create_gen_ox(int u, int v, int x, int y){
    vector<int> new_gen;
    int gen[200];
    bool check[200];
    memset(gen, -1, sizeof(gen));
    memset(check, false, sizeof(check));
    std::uniform_int_distribution<int> dist(1, city101 - 1);
   

    for(int i = x; i <= y; i++){
        gen[i] = par[u][i];
        check[par[u][i]] = true;
    }
    deque<int> pq;
    
    for(int i = y + 1; i < city101; i++)
         pq.push_back(par[v][i]);
    for(int i = 0; i <= y; i++)
        pq.push_back(par[v][i]);
    for(int i = y + 1; i < city101; i++){
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
    for(int i = 0; i < city101; i++)
        new_gen.push_back(gen[i]);

    return new_gen;

}

void cswap(vector<int>& child){
    std::uniform_int_distribution<int> dist(1, city101 - 1);
    
    int u = dist(g);
    int v = dist(g);
    while(v == u)
        v = dist(g);
    swap(child[u], child[v]);
    /*
    if(u > v)
        swap(u, v);
    for(int i = u; i <= u + (v - u + 1 ) / 2 - 1; i++)
        swap(child[i], child[v + u - i]);
    

    for(int i = 0; i < child.size(); i++){
        double rate = rate_dist(g);
        if(rate < rmp){
            int j = dist(g);
            swap(child[i], child[j]);
        }
    }
    */
}

double get_cost51(vector<int> x){
    double fit51(0);
    vector<int> pq;
    for(int u = 0; u < city101; u++)
        if(x[u] <= city51)
            pq.push_back(x[u]);

    for(int u = 1; u < city51; u++)
        fit51 += cost51[pq[u]][pq[u - 1]];
    fit51 += cost51[pq[0]][pq[city51 - 1]];
    return fit51;
}

double get_cost101(vector<int> x){
    double fit101(0);
    for(int u = 1; u < city101; u++)
        fit101 += cost101[x[u]][x[u - 1]];
    fit101 += cost101[x[0]][x[city101 - 1]];
    return fit101;
}

void get_type(){
    
    priority_queue<ii, vector<ii>, greater<ii>> pq51, pq101;
    for(int i = 1; i <= population; i++){
        int fit51(0), fit101(0);
        
        if(type[i] == 0)
        {
            pq101.push({1e9, i});
            fit51 = get_cost51(par[i]);
            pq51.push({fit51, i});
            continue;
        }
        if(type[i] == 1)
        {
            pq51.push({1e9, i});
            fit101 = get_cost101(par[i]);
            pq101.push({fit101, i});
            continue;
        }
        
        fit51 = get_cost51(par[i]);
        pq51.push({fit51, i});
        fit101 = get_cost101(par[i]);
        pq101.push({fit101, i});
        
    }
    int rank51[550], rank101[550], cnt(0);
    // Gán rank 51
    while(pq51.size()){
        ii u = pq51.top();
        cnt++;
        rank51[u.second] = cnt;
        pq51.pop();
    }
    
    // Gán rank 101
    cnt = 0;
    while(pq101.size()){
        ii v = pq101.top();
        cnt++;
        rank101[v.second] = cnt;
        pq101.pop();
    }
    for(int i = 1; i <= population; i++){
        if(rank51[i] == rank101[i])
            type[i] = dist(g) % 2;
        
        if(rank51[i] < rank101[i])
            type[i] = 0;
        
        if(rank51[i] > rank101[i])
            type[i] = 1;
        
        fitness[i] = 1.0/min(rank51[i], rank101[i]);
    }

}

void getnewpar(){

    priority_queue<ii, vector<ii>, greater<ii>> pq51, pq101;
    for(int i = 1; i <= population; i++){
        int fit51(0), fit101(0);
        
        if(type[i] == 0)
        {
            pq101.push({1e9, i});
            fit51 = get_cost51(par[i]);
            pq51.push({fit51, i});
            continue;
        }
        if(type[i] == 1)
        {
            pq51.push({1e9, i});
            fit101 = get_cost101(par[i]);
            pq101.push({fit101, i});
            continue;
        }
        
        fit51 = get_cost51(par[i]);
        pq51.push({fit51, i});
        fit101 = get_cost101(par[i]);
        pq101.push({fit101, i}); 
    }
    for(int i = 1; i <= population; i++){
        int fit51(0), fit101(0);
        
        if(parchil[i] == 0)
        {
            pq101.push({1e9, i + population});
            fit51 = get_cost51(child[i]);
            pq51.push({fit51, i + population});
            continue;
        }
        if(parchil[i] == 1)
        {
            pq51.push({1e9, i + population});
            fit101 = get_cost101(child[i]);
            pq101.push({fit101, i + population});
            continue;
        }
        
        fit51 = get_cost51(child[i]);
        pq51.push({fit51, i + population});
        fit101 = get_cost101(child[i]);
        pq101.push({fit101, i + population});
        
    }
    priority_queue<ii> pq;
    int rank51[550], rank101[550], cnt(0);
    // Gán rank 51
    while(pq51.size()){
        ii u = pq51.top();
        cnt++;
        rank51[u.second] = cnt;
        pq51.pop();
    }
    
    // Gán rank 101
    cnt = 0;
    while(pq101.size()){
        ii v = pq101.top();
        cnt++;
        rank101[v.second] = cnt;
        pq101.pop();
    }

    for(int i = 1; i <= population * 2; i++){
        fitness[i] = 1.0/min(rank51[i], rank101[i]);
        pq.push({fitness[i], i});
    }
    vector<int> newpar[550];
    map<int, int> mp;
    int count = 0;
    while(count < population){
        ii u = pq.top();
        if(u.second > population){
            if(mp[u.second] == 0){
                mp[u.second] = 1;
                count++;
                newpar[count] = child[u.second - population];
            }
        }
        else{
            if(mp[u.second] == 0){
                mp[u.second] = 1;
                count++;
                newpar[count] = par[u.second];
            }
        }
        pq.pop();
    }
    for(int i = 1; i <= population; i++)
        par[i] = newpar[i];
}
void getmax(int x){
    double fit51(1e9), fit101(1e9);
    for(int i = 1; i <= population; i++){
        fit51 = min(fit51, get_cost51(par[i]));
        fit101 = min(fit101, get_cost101(par[i]));
        
    }

    bestfit[x].first += fit51;
    bestfit[x].second += fit101;

}
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    freopen("cost_matrices.txt", "r", stdin);
    freopen("best_avg_mfea.txt", "w", stdout);

    cin >> city51;
    for(int i = 1; i <= city51; i++)
    for(int j = 1; j <= city51; j++)
        cin >> cost51[i][j];
    cin >> city101;
    for(int i = 1; i <= city101; i++)
    for(int j = 1; j <= city101; j++)
        cin >> cost101[i][j];

    for(int test = 1; test <= seed; test++){

        // tạo gen cha
        memset(type, -1, sizeof(type));

        vector<int> bg;
        for(int i = 1; i <= max(city51, city101); i++)
            bg.push_back(i);

        for(int i = 1; i <= population; i++){
            shuffle(bg.begin(), bg.end(), g);
            par[i] = bg;
        }

        getmax(1);
        
        for(int i = 2; i <= gen; i++){
        
            int cnt = 0;
            get_type();
            memset(parchil, -1, sizeof(parchil));

            for(int j = 1; j <= population / 2; j++){
                // tạo cha
                int par1, par2;

                // cha 1
                int u1 = dist(g), u2 = dist(g), u3 = dist(g);
                double um = max({fitness[u1], fitness[u2], fitness[u3]});
                if(fitness[u1] == um) par1 = u1;
                if(fitness[u2] == um) par1 = u2;
                if(fitness[u3] == um) par1 = u3;

                // cha 2
                int v1 = dist(g), v2 = dist(g), v3 = dist(g);
                double vm = max({fitness[v1], fitness[v2], fitness[v3]});
                if(fitness[v1] == vm) par2 = v1;
                if(fitness[v2] == vm) par2 = v2;
                if(fitness[v3] == vm) par2 = v3;
                 
                double rate = rate_dist(g);
                if(type[par1] == type[par2] or rate < rmp){
                    int x = dist(g);
                    int y = dist(g);
                    while(y < x)
                        y = dist(g);
                    cnt++;
                    child[cnt] = create_gen_ox(par1, par2, x, y);
                    parchil[cnt] = (rate_dist(g) < 0.5) ? type[par1] : type[par2];
                    cnt++;
                    child[cnt] = create_gen_ox(par2, par1, x, y);
                    parchil[cnt] = (rate_dist(g) < 0.5) ? type[par1] : type[par2];
                }
                else{
                    cnt++;
                    child[cnt] = par[par1];
                    cswap(child[cnt]);
                    parchil[cnt] = type[par1];
                    cnt++;
                    child[cnt] = par[par2];
                    cswap(child[cnt]);
                    parchil[cnt] = type[par2];
                }
            }

            getnewpar();
            getmax(i);
        }
        
        
    }
    
    for(int i = 1; i <= gen; i++)
        cout << bestfit[i].first * 1.0 / seed << " " << bestfit[i].second * 1.0 / seed << '\n';
}