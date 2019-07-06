#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

#define watch(x) cout<<(#x)<<"="<<(x)<<endl
#define mset(d,val) memset(d,val,sizeof(d))
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=a;i<b;i++)
#define fore(i,a,b) for(int i=a;i<=b;i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define PI 3.14159265358979323846264338327
#define INF 2e14
#define MOD 998244353
#define pqueue priority_queue
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef unsigned long long ull;
typedef tree<ll,null_type,less<ll>,rb_tree_tag,tree_order_statistics_node_update> pbds;

#define MAXN 100005

//Lazy Recursive ST start
class LazySegmentTree{
private:
	int size_;
	vector<ll> v,lazy;
	
	void update(int s, int e, ll val, int k, int l, int r){
		push(k,l,r);
		if(r < s || e < l) return;
		if(s <= l && r <= e){
			lazy[k] = val;
			push(k, l, r);
		}
		else{
			update(s, e, val, k*2, l, (l+r)>>1);
			update(s, e, val, k*2+1, ((l+r)>>1)+1, r);
			v[k] = merge(v[k*2], v[k*2+1]);
		}
	}
	
	ll query(int s, int e, int k, int l, int r){
		push(k,l,r);
		if(r < s || e < l) return 0;
		if(s <= l && r <= e) return v[k];
		ll lc = query(s, e, k*2, l, (l+r)>>1);
		ll rc = query(s, e, k*2+1, ((l+r)>>1)+1, r);
		return merge(lc, rc);
	}
 
public:
	LazySegmentTree(): v(vector<ll>()), lazy(vector<ll>()) {};
	LazySegmentTree(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
		lazy.resize(size_*4);
	}
	inline void push(int k, int l, int r){
		if(lazy[k]!=0){
			v[k]+=lazy[k];
			if(l!=r){
				lazy[k*2]+=lazy[k];
				lazy[k*2+1]+=lazy[k];
			}
			lazy[k]=0;
		}
	}
	inline ll merge(ll x, ll y){
		return x+y;
	}
	inline void update(int l, int r, ll val){
		update(l, r, val, 1, 0, size_-1);
	}
	inline ll query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Lazy recursive ST end

//DSU start
struct DSU{
	struct node{ int p; ll sum; };
	vector<node> dsu;
	DSU(int n){ dsu.resize(n);
		forn(i,0,n){ dsu[i].p=i; dsu[i].sum=0;}
	}
	int rt(int u){ return (dsu[u].p==u) ? u : dsu[u].p=rt(dsu[u].p); }
	bool sameset(int u, int v){ return rt(u)==rt(v); }
	void merge(int u, int v){
		u = rt(u); v = rt(v);
		if(u == v) return;
		if(rand()&1) swap(u,v);
		dsu[v].p = u;
		//dsu[u].sum += dsu[v].sum;
	}
	//ll getstat(int u){ return dsu[rt(u)].sum; }
	//void setstat(int u, ll val){ dsu[rt(u)].sum = val; }
};
//DSU end

//Kruskal start
int n,m;
vector<pair<ll,ii>> edge;
vector<pair<ii,ll>> mst;
int cnt=0;
ll sumw=0;

void kruskal(){
	DSU dsu(n);
	sort(edge.begin(),edge.end());
	
	forn(i,0,m){
		int u=edge[i].S.F, v=edge[i].S.S; ll w=edge[i].F;
		if(dsu.sameset(u,v)) continue;
		mst.pb({{u,v},w});
		dsu.merge(u,v);
		sumw+=w;
		cnt++;
		if(cnt>=n-1) break;
	}
}
//Kruskal end

//Dijkstra start
vii adj[MAXN];
ll dist[MAXN];

void dijkstra(int src){
	pqueue<ii,vii,greater<ii>> q;
	forn(i,0,n)	dist[i]=INF;
	dist[src]=0;
	q.push({0,src});
	
	while(!q.empty()){
		int u=q.top().S; q.pop();
		for(auto tmp: adj[u]){
			int v=tmp.F; ll w=tmp.S;
			if(dist[v]>dist[u]+w){
				dist[v]=dist[u]+w;
				q.push({dist[v],v});
			}
		}
	}
}
//Dijkstra end

//Floyd start
ll dist[MAXN][MAXN];
void Floyd(){
	forn(i,0,n) forn(j,0,n) dist[i][j]=adj[i][j];
	forn(k,0,n) forn(i,0,n) forn(j,0,n){
		if(dist[i][j]>dist[i][k]+dist[k][j])
			dist[i][j]=dist[i][k]+dist[k][j];
	}
}
//Floyd end

//HLD start
#define LG 19

vi adj[MAXN];
int in[MAXN],out[MAXN],rin[MAXN];
int prt[MAXN][LG];
int sz[MAXN],dep[MAXN];
int top[MAXN];

void dfs_sz(int u, int p){
	sz[u]=1; 
	prt[u][0]=p;

	if(adj[u][0]==p && adj[u].size()>1) swap(adj[u][0],adj[u][1]);
	
	for(int &v: adj[u]){
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_sz(v,u);
		sz[u]+=sz[v];
		if(sz[v]>sz[adj[u][0]]) swap(v,adj[u][0]);
	}
}

int tmr=0;
void dfs_hld(int u, int p){
	in[u]=tmr++;
	for(int v: adj[u]){
		if(v==p) continue;
		top[v] = (v==adj[u][0]) ? top[u] : v;
		dfs_hld(v,u);
	}
	out[u]=tmr;
}

ll Query(int u,int v){
	ll ans=0;
	while(top[u]!=top[v]){
		if(dep[top[u]]<dep[top[v]]) swap(u,v);
		ans=ans + st.query(in[top[u]],in[u]);
		u=prt[top[u]];
	}
	
	if(dep[u]<dep[v]) swap(u,v);
	return ans + st.query(in[v],in[u]);
}

st.update(in[u],in[u],w);
dfs_sz(0,0);
//HLD end

//Binary parent start
int goup(int u, int h){
	for(int i=LG;i>=0;i--){
		if(i&(1<<h)) u=prt[u][i];
	}
	return u;
}
//Binary parent end

//Sparse Table start: O(1) Min Query example
struct SparseTable{
	ll spt[MAXN][LG];
	int lg[MAXN+1];
	
	ll merge(ll x,ll y){
		return min(x,y);
	}
	
	SparseTable(int n, ll arr[]){
		lg[1]=0;
		fore(i,2,n)	lg[i]=lg[i/2]+1;
		
		forn(i,0,n) spt[i][0] = arr[i];
		fore(j,1,LG)
			for(int i=0; i+(1<<j)<=n; i++)
				spt[i][j] = merge(spt[i][j-1], spt[i+(1<<(j-1))][j-1]);
	}
	
	ll query(int l,int r){
		int len=lg[r-l+1];		
		return merge(spt[l][len],spt[r-(1<<len)+1][len]);
	}	
};

#define LG 25
//Sparse Table end

//LCA start
#define LG 18

int depth[MAXN],prt[LG+1][MAXN];
mset(prt,-1);

void dfs(int u, int p)
{
	prt[0][u]=p;
	forn(j,1,LG){
		if(prt[u][j-1]!=-1) prt[u][j]=prt[prt[u][j-1]][j-1];
	}
	for(int v:adj[u])
	{
		if(v==p) continue;
		depth[v]=depth[u]+1;
		dfs(v,u);
	}
}

int lca(int u, int v)
{
	if(depth[u]>depth[v]) swap(u,v);
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[i][v]!=-1&&depth[prt[i][v]]>=depth[u])
		{
			v=prt[i][v];
		}
	}
	if(u==v) return u;
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[i][v]!=-1&&prt[i][v]!=prt[i][u])
		{
			v=prt[i][v]; u=prt[i][u];
		}
	}
	return prt[0][u];
}
//LCA end

//Iterative ST start
ll t[2*MAXN];

void build(){
	for(int i=n-1;i>0;i--) 	t[i]=t[2*i]+t[2*i+1];	
}

void update(int p,int val){
	for(t[p+=n]=val;p>1;p/=2)	t[p/2]=t[p]+t[p^1];
}

ll query(int l,int r){
	r++;
	ll sum=0;
	for(l+=n,r+=n;l<r;l/=2,r/=2){
		if(l&1)	sum+=t[l++];
		if(r&1)	sum+=t[--r];
	}
	return sum;
}
//Iterative ST end
