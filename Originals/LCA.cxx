#include <bits/stdc++.h>
using namespace std;

#define debug(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)
#define timeS clock_t t1,t2; t1=clock();
#define timeE t2=clock(); debug(((float)t2-(float)t1)/CLOCK_PER_SEC);
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=a;i<b;i++)
#define fore(i,a,b) for(int i=a;i<=b;i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define PI 3.14159265358979323846264338327
#define INF 2000000000
#define MOD 998244353
#define pqueue priority_queue
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef unsigned long long ull;

#define MAXN 100005
#define MAXL 18

int n,m;
vi adj[MAXN];
int prt[MAXN][MAXL],depth[MAXN];

void dfs(int u,int p){
	depth[u]=depth[p]+1;
	prt[u][0]=p;
	
	for(auto v:adj[u]){
		if(v!=p)	dfs(v,u);
	}
}

void initLCA(){
	forn(j,1,MAXL){
		fore(u,1,n){
			if(prt[u][j-1]!=-1)	prt[u][j]=prt[ prt[u][j-1] ][j-1];
		}
	}
}

int lca(int u,int v){
	if(depth[u]<depth[v])	swap(u,v);
	int diff=depth[u]-depth[v];
	
	forn(i,0,MAXL){
		if(1<<i&diff)	u=prt[u][i];
	}
	
	if(u==v)	return u;
	
	for(int i=MAXL-1;i>=0;i--){
		if(prt[u][i]!=prt[v][i]){
			u=prt[u][i];
			v=prt[v][i];
		}
	}
	
	return prt[u][0];
}


int main()
{
	//ios_base::sync_with_stdio(0); cin.tie(0);
	memset(prt,-1,sizeof(prt));
	
	cin>>n;
	
	forn(i,0,n-1){
		int u,v;	cin>>u>>v;
		adj[u].pb(v);
		adj[v].pb(u);
	}
	
	dfs(1,-1);
	initLCA();
	
	while(1){
		cout<<"Select nodes: ";
		int u,v;	cin>>u;
		if(u==-1)	return 0;
		cin>>v;
		cout<<"LCA: "<<lca(u,v)<<'\n';
	}
	
	return 0;
}
