#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

#define watch(x) cout<<(#x)<<"="<<(x)<<'\n'
#define mset(d,val) memset(d,val,sizeof(d))
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=a;i<b;i++)
#define fore(i,a,b) for(int i=a;i<=b;i++)
#define pb push_back
#define F first
#define S second
#define INF ll(1e18)
#define MOD 998244353
#define pqueue priority_queue
#define fbo find_by_order
#define ook order_of_key
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef long double ld;
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> pbds;

#define MAXN 100005

//Lazy Recursive ST start
class LazySegmentTree{
private:
	int size_;
	vector<ll> v,lazy;
	
	void update(int s, int e, ll val, int k, int l, int r){
		push(k, l, r);
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
		push(k, l, r);
		if(r < s || e < l) return 0; //dummy value
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
	void reset(){
		v.assign(size_*4,0);
		lazy.assign(size_*4,0);
	}
	inline void push(int k, int l, int r){
		if(lazy[k]!=0){
			v[k]+=(r-l+1)*lazy[k];	//remember to consider the range!
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

//Lazy recursive ST with struct start
struct Node{
	ll sum,mn,mx;
	Node(){sum=mn=mx=0;}
	Node(ll _s,ll _mn,ll _mx){ sum=_s; mn=_mn; mx=_mx; }
};

class LazySegmentTreeNode{
private:	
	int size_;
	vector<Node> v,lazy;
	
	void update(int s, int e, Node val, int k, int l, int r){
		push(k, l, r);
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
	
	Node query(int s, int e, int k, int l, int r){
		push(k, l, r);
		if(r < s || e < l) return Node(0,INF,-1); //dummy value
		if(s <= l && r <= e) return v[k];
		Node lc = query(s, e, k*2, l, (l+r)>>1);
		Node rc = query(s, e, k*2+1, ((l+r)>>1)+1, r);
		return merge(lc, rc);
	}
 
public:
	LazySegmentTreeNode(): v(vector<Node>()), lazy(vector<Node>()) {};
	LazySegmentTreeNode(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
		lazy.resize(size_*4);
	}
	//void reset(){}
	inline void push(int k, int l, int r){
		if(lazy[k]!=Node(0,0,0)){
			v[k].sum+=(r-l+1)*lazy[k].sum;	//remember to consider the range!
			v[k].mn+=lazy[k].mn;
			v[k].mx+=lazy[k].mx;
			if(l!=r){
				lazy[k*2].sum+=lazy[k].sum;	lazy[k*2+1].sum+=lazy[k].sum;
				lazy[k*2].mn+=lazy[k].mn;	lazy[k*2+1].mn+=lazy[k].mn;
				lazy[k*2].mx+=lazy[k].mx;	lazy[k*2+1].mx+=lazy[k].mx;
			}
			lazy[k]=Node(0,0,0);
		}
	}
	inline Node merge(Node x, Node y){
		Node tmp;
		tmp.sum = x.sum + y.sum;
		tmp.mn = min(x.mn, y.mn);
		tmp.mx = max(x.mx, y.mx);
		return tmp;
	}
	inline void update(int l, int r, Node val){
		update(l, r, val, 1, 0, size_-1);
	}
	inline Node query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Lazy recursive ST with struct end

//Point recursive ST start
class PointSegmentTree{
private:
	int size_;
	vector<ll> v;
	
	void update(int p, ll val, int k, int l, int r)
	{
		if(p < l || r < p) return;
		if(l == r){
			v[k]=val;	//modification
			return;
		}
		int mid = (l+r)>>1;
		update(p, val, k*2, l, mid);
		update(p, val, k*2+1, mid+1, r);
		v[k] = merge(v[k*2], v[k*2+1]);
	}
	
	ll query(int s, int e, int k, int l, int r)
	{
		if(e < l || r < s) return 0; //dummy value
		if(s <= l && r <= e) return v[k];
		int mid = (l+r)>>1;
		ll lc = query(s, e, k*2, l, mid);
		ll rc = query(s, e, k*2+1, mid+1, r);
		return merge(lc, rc);
	}
	
public:
	PointSegmentTree(): v(vector<ll>()) {};
	PointSegmentTree(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
	}
	//void reset(){}
	inline ll merge(ll x, ll y){
		return x+y;
	}
	inline void update(int p, ll val){
		update(p, val, 1, 0, size_-1);
	}
	inline ll query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Point recursive ST end

//Point recursive ST with struct start
struct Node{
	ll sum,mn,mx;
	Node(){sum=mn=mx=0;}
	Node(ll _s,ll _mn,ll _mx){ sum=_s; mn=_mn; mx=_mx; }
};

class PointSegmentTreeNode{
private:
	int size_;
	vector<Node> v;
	
	void update(int p, Node val, int k, int l, int r)
	{
		if(p < l || r < p) return;
		if(l == r){
			v[k].sum += val.sum; //modifications
			v[k].mx = val.mx;
			v[k].mn = val.mn;
			return;
		}
		int mid = (l+r)>>1;
		update(p, val, k*2, l, mid);
		update(p, val, k*2+1, mid+1, r);
		v[k] = merge(v[k*2], v[k*2+1]);
	}
	
	Node query(int s, int e, int k, int l, int r)
	{
		if(e < l || r < s) return Node(0,INF,-1); //dummy value
		if(s <= l && r <= e) return v[k];
		int mid = (l+r)>>1;
		Node lc = query(s, e, k*2, l, mid);
		Node rc = query(s, e, k*2+1, mid+1, r);
		return merge(lc, rc);
	}
	
public:
	PointSegmentTreeNode(): v(vector<Node>()) {};
	PointSegmentTreeNode(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
	}
	//void reset(){}
	inline Node merge(Node x, Node y){
		Node tmp;
		tmp.sum = x.sum + y.sum;
		tmp.mn = min(x.mn, y.mn);
		tmp.mx = max(x.mx, y.mx);
		return tmp;
	}
	inline void update(int p, ll _sm=0, ll _mn=INF, ll _mx=0){
		update(p, Node(_sm,_mn,_mx), 1, 0, size_-1);
	}
	inline Node query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Point recursive with struct ST end

//Point iterative ST start
struct IterSegmentTree{
	vector<ll> t;
	
	IterSegmentTree: t(vector<ll>()) {}
	IterSegmentTree(int _n){
		t.resize(_n*2);
	}
	
	void build(){
		for(int i=n-1; i>0; i--) t[i]=t[i<<1]+t[i<<1|1];
	}
	
	void update(int p, int val){
		for(t[p+=n]=value; p>1; p>>=1) t[p>>1]=t[p]+t[p^1];
	}
	
	int query(int l, int r){
		int res=0;
		for(l+=n,r+=n; l<r; l>>=1,r>>=1){
			if(l&1) res+=t[l++];
			if(r&1) res+=t[--r];
		}
		return res;
	}
};

forn(i,0,n) cin>>t[n+i];
//Point iterative ST end

//FenwickRange start
struct FenwickRange
{
	vector<ll> fw,fw2;
    int siz;
    FenwickRange(int N)
    {
        fw.assign(N+1,0);
        fw2.assign(N+1,0);
        siz = N+1;
    }
    void reset(int N)
    {
		fw.assign(N+1,0);
        fw2.assign(N+1,0);
        siz = N+1;
	}
    void add(int l, int r, ll val) //[l,r] + val
    {   
		l++; r++;
        for(int tl=l; tl<siz; tl+=(tl&(-tl)))
        {
            fw[tl]+=val, fw2[tl]-=val*ll(l-1);
        }
        for(int tr=r+1; tr<siz; tr+=(tr&(-tr)))
        {
            fw[tr]-=val, fw2[tr]+=val*ll(r);
        }
    }
    ll sum(int r) //[1,r]
    {                         
        ll res=0;
        for(int tr=r; tr; tr-=(tr&(-tr)))
        {
            res+=fw[tr]*ll(r)+fw2[tr];
        }
        return res;
    }
    ll query(int l, int r)
    {
		l++; r++;
		if(r<l) return 0;
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
	void modify(int p, ll val)
	{
		add(p, val-query(p,p));
	}
};
//FenwickRange end

//Fenwick Tree (FenwickPoint) start
struct FenwickPoint{
    vector<ll> fw;
    int siz;
    FenwickPoint(int N)
    {
        fw.assign(N+1,0);
        siz = N+1;
    }
    void reset(int N)
    {
		fw.assign(N+1,0);
        siz = N+1;
	}
	void add(int p, ll val)
	{
		for(p++; p<=siz; p+=(p&(-p)))
		{
			fw[p]+=val;
		}
	}
	ll sum(int p)
	{
		ll res=0;
		for(; p; p-=(p&(-p)))
		{
			res+=fw[p];
		}
		return res;
	}
	ll query(int l, int r)
    {
		l++; r++;
		if(r<l) return 0;
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
	void modify(int p, ll val)
	{
		add(p, val-query(p,p));
	}
};
//Fenwick Tree (FenwickPoint) end

//Randomizer start
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int>(1,6)(rng);
uniform_int_distribution<> dis(1,6);
	
	Examples:
	cout<<rng()<<'\n';
	cout<<dis(rng)<<'\n';

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
//Randomizer end

//KMP/prefix function start
vector<int> prefix_function(string &Z){
	int n=(int)Z.length();
	vector<int> F(n);
	F[0]=0;
	forn(i,1,n){
		int j=F[i-1];
		while(j>0 && Z[i]!=Z[j]){
			j=F[j-1];
        }
		if(Z[i]==Z[j]) j++;
		F[i]=j;
	}
	return F;
}
//KMP/prefix function end

//Trie start
struct TrieNode{
	int next[26];
	bool leaf = false;
	
    TrieNode(){fill(begin(next), end(next), -1);}
};

vector<TrieNode> Trie(1);

void addstring(const string &s){
    int v = 0;
    for(char ch : s){
		int c = ch - 'a';
		if(Trie[v].next[c] == -1){
			Trie[v].next[c] = Trie.size();
			Trie.emplace_back();
		}
		v = Trie[v].next[c];
	}
	Trie[v].leaf = true;
}
//Trie end

//DSU start
struct DSU{
	struct node{ int p; ll sz; };
	vector<node> dsu; int cc;
	DSU(int n){ dsu.resize(n);
		forn(i,0,n){ cc=n; dsu[i].p=i; dsu[i].sz=1;}
	}
	int rt(int u){ return (dsu[u].p==u) ? u : dsu[u].p=rt(dsu[u].p); }
	bool sameset(int u, int v){ return rt(u)==rt(v); }
	void merge(int u, int v){
		u = rt(u); v = rt(v);
		if(u == v) return;
		if(dsu[u].sz < dsu[v].sz) swap(u,v);
		dsu[v].p = u;
		dsu[u].sz += dsu[v].sz;
		cc--;
	}
	//ll get(int u){ return dsu[rt(u)].sz; }
	//void set(int u, ll val){ dsu[rt(u)].sz = val; }
};
//DSU end

//Kruskal start
int n,m;
vector<pair<ll,ii>> edges;
vector<pair<ii,ll>> mst;
ll sumw=0;

void kruskal()
{
	DSU dsu(n);
	sort(edges.begin(),edges.end());
	sumw=0;
	
	forn(i,0,edges.size()){
		int u=edges[i].S.F, v=edges[i].S.S; ll w=edges[i].F;
		if(dsu.sameset(u,v)) continue;
		mst.pb({{u,v},w});
		dsu.merge(u,v);
		sumw+=w;
		if(dsu.cc<=1) break;
	}
}
//Kruskal end

//Dijkstra start
vii adj[MAXN];
ll dist[MAXN];

void dijkstra(int src)
{
	pqueue<ii,vii,greater<ii>> q;
	forn(i,0,n)	dist[i]=INF;
	dist[src]=0;
	q.push({dist[src],src});
	
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
	forn(i,0,n) forn(j,0,n) dist[i][j] = (adj[i][j]==0 ? INF : adj[i][j]);
	forn(k,0,n) forn(i,0,n) forn(j,0,n)
		dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}
//Floyd end

//Euler path start
int in[MAXN],out[MAXN];
int tmr=-1;

void dfs_euler(int u, int p){
	in[u]=++tmr;
	for(int v: adj[u]){
		if(v==p) continue;
		dfs_euler(v,u);
	}
	out[u]=tmr;
}
//Euler path end

//HLD (Heavy-light decomposition) start
#define LG 19

int in[MAXN],out[MAXN];
int tmr=-1;
int prt[MAXN];
int sz[MAXN],dep[MAXN];
int top[MAXN];

void dfs_sz(int u, int p){
	sz[u]=1;
	prt[u]=p;

	if(adj[u][0]==p && adj[u].size()>1) swap(adj[u][0],adj[u][1]);
	
	for(int &v: adj[u]){
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_sz(v,u);
		sz[u]+=sz[v];
		if(sz[v]>sz[adj[u][0]]) swap(v,adj[u][0]);
	}
}

void dfs_hld(int u, int p){
	in[u]=++tmr;
	for(int v: adj[u]){
		if(v==p) continue;
		top[v] = (v==adj[u][0]) ? top[u] : v;
		dfs_hld(v,u);
	}
	out[u]=tmr;
}

ll merge_hld(ll x, ll y){ return x+y; }
ll Query(int u, int v){
	ll ans=0;
	while(top[u]!=top[v]){
		if(dep[top[u]]<dep[top[v]]) swap(u,v);
		ans=merge_hld(ans,st.query(in[top[u]],in[u]));
		u=prt[top[u]];
	}
	
	if(dep[u]<dep[v]) swap(u,v);
	return merge_hld(ans,st.query(in[v],in[u]));
}

st.update(in[u],in[u],w);
dfs_sz(0,-1);
dfs_hld(0,-1);
//HLD (Heavy-light decomposition) end

//Sparse Table start: O(1) Min Query
#define LG 25

int lg[MAXN+1];
ll spt[MAXN][LG+1];

struct SparseTable
{	
	ll merge(ll x, ll y){
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
//Sparse Table end

//LCA O(log n) query start
#define LG 11

int dep[MAXN],prt[MAXN][LG];
mset(prt,-1); mset(dep,0);

void dfs_lca(int u, int p)
{
	prt[u][0]=p;
	forn(j,1,LG){
		if(prt[u][j-1]!=-1) prt[u][j]=prt[prt[u][j-1]][j-1];
	}
	for(int v:adj[u])
	{
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_lca(v,u);
	}
}

int lca(int u, int v)
{
	if(dep[u]>dep[v]) swap(u,v);
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[v][i]!=-1&&dep[prt[v][i]]>=dep[u])
		{
			v=prt[v][i];
		}
	}
	if(u==v) return u;
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[v][i]!=-1&&prt[v][i]!=prt[u][i])
		{
			v=prt[v][i]; u=prt[u][i];
		}
	}
	return prt[u][0];
}
//LCA O(log n) query end

//Binary parent start
int goup(int u, int h){
	for(int i=LG-1;i>=0;i--){
		if(h&(1LL<<i)) u=prt[u][i];
	}
	return u;
}
//Binary parent end

//LCA O(1) query start
vi adj[MAXN];
int lg[MAXN+1];
ll spt[MAXN][LG+1];
int in[MAXN],out[MAXN],dep[MAXN];
vi euler;
int tmr=-1;

struct SparseTableLCA
{	
	ll merge(ll x, ll y){
		return (dep[x]<dep[y] ? x:y);
	}
	
	SparseTableLCA(){}	
	SparseTableLCA(vi arr){
		int N=arr.size();
		lg[1]=0;
		fore(i,2,N)	lg[i]=lg[i/2]+1;
		
		forn(i,0,N) spt[i][0] = arr[i];
		fore(j,1,LG)
			for(int i=0; i+(1<<j)<=N; i++)
				spt[i][j] = merge(spt[i][j-1], spt[i+(1<<(j-1))][j-1]);
	}
	
	ll query(int l,int r){
		int len=lg[r-l+1];		
		return merge(spt[l][len],spt[r-(1<<len)+1][len]);
	}	
}lcast;

void dfs_lca(int u, int p){
	in[u]=++tmr;
	euler.pb(u);
	for(int v: adj[u]){
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_lca(v,u);
		euler.pb(u);
	}
	out[u]=tmr;
}

int lca(int u, int v){
	int l=in[u],r=in[v];
	if(l>r) swap(l,r);
	return lcast.query(l,r);
}	

//in main()
dfs_lca(0,-1);
lcast=SparseTableLCA(euler);

//LCA O(1) query end

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

//Combi/Maths start
vector<ll> fact,ifact,inv,pow2;
ll add(ll a,ll b)
{
	a%=MOD; b%=MOD;
	a+=b;a%=MOD;
	if(a<0) a+=MOD;
	return a;
}
ll mult(ll a, ll b)
{
	a%=MOD; b%=MOD;
	ll ans=(a*b)%MOD;
	if(ans<0) ans+=MOD;
	return ans;
}
ll pw(ll a, ll b)
{
	ll r=1;
	while(b){
		if(b&1) r=mult(r,a);
		a=mult(a,a);
		b>>=1;
	}
	return r;
}
ll inverse(ll a)
{
	return pw(a,MOD-2);
}
ll choose(ll a, ll b)
{
	if(a<b) return 0;
	if(b==0) return 1;
	if(a==b) return 1;
	return mult(fact[a],mult(ifact[b],ifact[a-b]));
}
void init(ll _n)
{
	fact.clear(); ifact.clear(); inv.clear(); pow2.clear();
	fact.resize(_n+1); ifact.resize(_n+1); inv.resize(_n+1); pow2.resize(_n+1);
	pow2[0]=1; ifact[0]=1; fact[0]=1;
	for(int i=1;i<=_n;i++){
		pow2[i]=add(pow2[i-1],pow2[i-1]);
		fact[i]=mult(fact[i-1],i);
	}
	ifact[_n] = inverse(fact[_n]);
	for(int i=_n-1;i>=1;i--){
	    ifact[i] = mult(ifact[i+1], i+1);
	}
	for(int i=1;i<=_n;i++){
	    inv[i] = mult(fact[i-1], ifact[i]);
	}
}
void getpf(vector<ii>& pf, ll n)
{
	for(ll i=2; i*i<=n; i++)
	{
		int cnt=0;
		while(n%i==0){
			n/=i; cnt++;
		}
		if(cnt>0) pf.pb({i,cnt});
	}
	if(n>1) pf.pb({n,1});
}
//Combi/Maths end

//Matrix start
struct Matrix{
	ll a[DIM][DIM];
	ll *operator[](ll r){ return a[r]; }

	Matrix(int x = 0){
		memset(a, 0, sizeof(a));
		if(x) for(int i = 0; i < DIM; i++) a[i][i] = x;
	}
} const I(1);

Matrix operator*(Matrix A, Matrix B){
	const ll MOD2 = ll(MOD)*MOD;
	Matrix C;
	for(int i = 0; i < DIM; i++){
		for(int j = 0; j < DIM; j++){
			ll w = 0;
			for(int k = 0; k < DIM; k++){
				w += ll(A[i][k]) * B[k][j];
				if (w >= MOD2) w -= MOD2;
			}
			C[i][j] = w % MOD;
		}
	}
	return C;
}

Matrix operator^(Matrix A, ll b){
	Matrix R = I;
	for(; b > 0; b /= 2){
		if(b&1) R = R*A;
		A = A*A;
	}
	return R;
}
//Matrix end

//Number Theory NT start
vector<ll> primes;
vector<bool> prime;
vector<ll> totient;
vector<ll> sumdiv;
vector<ll> bigdiv;
void Sieve(ll n)
{
	prime.assign(n+1, 1);
	prime[1] = false;
	for(ll i = 2; i <= n; i++)
	{
		if(prime[i])
		{
			primes.pb(i);
			for(ll j = i*2; j <= n; j += i)
			{
				prime[j] = false;
			}
		}
	}
}

ll phi(ll x)
{
	map<ll,ll> pf;
	ll num = 1; ll num2 = x;
	for(ll i = 0; primes[i]*primes[i] <= x; i++)
	{
		if(x%primes[i]==0)
		{
			num2/=primes[i];
			num*=(primes[i]-1);
		}
		while(x%primes[i]==0)
		{
			x/=primes[i];
			pf[primes[i]]++;
		}
	}
	if(x>1)
	{
		pf[x]++; num2/=x; num*=(x-1);
	}
	x = 1;
	num*=num2;
	return num;
}

bool isprime(ll x)
{
	if(x==1) return false;
	for(ll i = 0; primes[i]*primes[i] <= x; i++)
	{
		if(x%primes[i]==0) return false;
	}
	return true;
}

void SievePhi(ll n)
{
	totient.resize(n+1);
	for (int i = 1; i <= n; ++i) totient[i] = i;
	for (int i = 2; i <= n; ++i)
	{
		if (totient[i] == i)
		{
			for (int j = i; j <= n; j += i)
			{
				totient[j] -= totient[j] / i;
			}
		}
	}
}

void SieveSumDiv(ll n)
{
	sumdiv.resize(n+1);
	for(int i = 1; i <= n; ++i)
	{
		for(int j = i; j <= n; j += i)
		{
			sumdiv[j] += i;
		}
	}
}

ll getPhi(ll n)
{
	return totient[n];
}

ll getSumDiv(ll n)
{
	return sumdiv[n];
}

ll modpow(ll a, ll b, ll mod)
{
	ll r = 1;
	if(b < 0) b += mod*100000LL;
	while(b)
	{
		if(b&1) r = (r*a)%mod;
		a = (a*a)%mod;
		b>>=1;
	}
	return r;
}

ll inv(ll a, ll mod)
{
	return modpow(a, mod - 2, mod);
}

ll invgeneral(ll a, ll mod)
{
	ll ph = phi(mod);
	ph--;
	return modpow(a, ph, mod);
}

void getpf(vector<ii>& pf, ll n)
{
	for(ll i = 0; primes[i]*primes[i] <= n; i++)
	{
		int cnt = 0;
		while(n%primes[i]==0)
		{
			n/=primes[i]; cnt++;
		}
		if(cnt>0) pf.pb(ii(primes[i], cnt));
	}
	if(n>1)
	{
		pf.pb(ii(n, 1));
	}
}

void getDiv(vector<ll>& div, vector<ii>& pf, ll n = 1, int i = 0)
{
	ll x, k;
	if(i >= pf.size()) return;
	x = n;
	for(k = 0; k <= pf[i].S; k++)
	{
		if(i==int(pf.size())-1) div.pb(x);
		getDiv(div, pf, x, i + 1);
		x *= pf[i].F;
	}
}
//End Number Theory NT

//Sqrt Decomposition/Mo's algorithm start
int BS;
struct query{
	int l,r,id;
};
bool cmp(query a, query b){
	if(a.l/BS != b.l/BS) return a.l/BS<b.l/BS;
	return a.r<b.r;
}

int n,Q;
int cur=0;
ll a[30005];
ll cnt[1000005];
query q[200005];
ll ans[200005];

void add(int p){
	if(!cnt[a[p]]) cur++;
	cnt[a[p]]++;
}

void remove(int p){
	cnt[a[p]]--;
	if(!cnt[a[p]]) cur--;
}

mset(cnt,0);
	
BS=sqrt(n);
forn(i,0,n) cin>>a[i];
cin>>Q;
forn(i,0,Q){
	int l,r; cin>>l>>r; l--; r--;
	q[i].l=l; q[i].r=r;
	q[i].id=i;
}

sort(q,q+Q,cmp);

int mo_l=0, mo_r=-1;
forn(i,0,Q){
	int L=q[i].l, R=q[i].r;
	
	while(mo_l<L) remove(mo_l++);
	while(mo_l>L) add(--mo_l);
	while(mo_r<R) add(++mo_r);
	while(mo_r>R) remove(mo_r--);
	
	ans[q[i].id]=cur;
}
//Sqrt decomposition/Mo's algorithm end

//Convex Hull Dynamic short start (CHT)
struct Line{
	mutable ll m,b,p;
	bool operator<(const Line& o) const { return m < o.m; }
	bool operator<(ll x) const { return p < x; }
};

struct ConvexHullDynamic: multiset<Line, less<>> {
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	const ll inf = LLONG_MAX;
	bool Max = 1;
	
	ll div(ll a, ll b) { // floored division
		return a / b - ((a ^ b) < 0 && a % b); }
	bool isect(iterator x, iterator y) {
		if (y == end()) { x->p = inf; return false; }
		if (x->m == y->m) x->p = x->b > y->b ? inf : -inf;
		else x->p = div(y->b - x->b, x->m - y->m);
		return x->p >= y->p;
	}
	void addline(ll m, ll b) {
		if(!Max) { m=-m; b=-b; }
		auto z = insert({m, b, 0}), y = z++, x = y;
		while (isect(y, z)) z = erase(z);
		if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}
	ll query(ll x) {
		//if(empty()) return 0;
		auto l = *lower_bound(x);
		return (l.m * x + l.b)*(Max ? 1 : -1);
	}
};
//Convex Hull Dynamic short end (CHT)

//Convex Hull Dynamic short 2 start
const ll is_query = -(1LL<<62);

struct Line{
    ll m,b;
    mutable function<const Line*()> succ;
    bool operator<(const Line& rhs) const{
        if(rhs.b != is_query) return m < rhs.m;
        const Line* s = succ();
        if(!s) return 0;
        ll x = rhs.m;
        return 1.0L * b - s->b < 1.0L * (s->m - m) * x;
    }
};

struct ConvexHullDynamic: public multiset<Line>{ //will maintain upper hull for maximum
	bool Max = 1;
	
    bool bad(iterator y){
        auto z = next(y);
        if(y == begin()){
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if(z == end()) return y->m == x->m && y->b <= x->b;
        return (x->b - y->b)*1.0L*(z->m - y->m) >= (y->b - z->b)*1.0L*(y->m - x->m);
    }
    void addline(ll m, ll b){
		if(!Max) { m=-m; b=-b; }
        auto y = insert({m,b});
        y->succ = [=] { return next(y)==end() ? 0 : &*next(y); };
        if(bad(y)) { erase(y); return; }
        while(next(y)!=end() && bad(next(y))) erase(next(y));
        while(y != begin() && bad(prev(y))) erase(prev(y));
    }
    ll query(ll x){
		//if(empty()) return 0;
        auto l = *lower_bound((Line){x,is_query});
        return (l.m * x + l.b)*(Max ? 1 : -1);
    }
};
//Convex Hull Dynamic short 2 end

//Convex Hull Dynamic long start
class ConvexHullDynamic {
	typedef long long coef_t;
	typedef long long coord_t;
	typedef long long val_t;

private:
	struct Line {
		coef_t a, b;
		double xLeft;

		enum Type {
			line, maxQuery, minQuery
		} type;
		coord_t val;

		explicit Line(coef_t aa = 0, coef_t bb = 0) :
				a(aa), b(bb), xLeft(-INF), type(Type::line), val(0) {
		}
		val_t valueAt(coord_t x) const {
			return a * x + b;
		}
		friend bool areParallel(const Line& l1, const Line& l2) {
			return l1.a == l2.a;
		}
		friend double intersectX(const Line& l1, const Line& l2) {
			return areParallel(l1, l2) ?
					INF : 1.0 * (l2.b - l1.b) / (l1.a - l2.a);
		}
		bool operator<(const Line& l2) const {
			if (l2.type == line)
				return this->a < l2.a;
			if (l2.type == maxQuery)
				return this->xLeft < l2.val;
			if (l2.type == minQuery)
				return this->xLeft > l2.val;

			return 0;
		}
	};

private:
	bool isMax;
	std::set<Line> hull;

private:
	bool hasPrev(std::set<Line>::iterator it) {
		return it != hull.begin();
	}
	bool hasNext(std::set<Line>::iterator it) {
		return it != hull.end() && std::next(it) != hull.end();
	}
	bool irrelevant(const Line& l1, const Line& l2, const Line& l3) {
		return intersectX(l1, l3) <= intersectX(l1, l2);
	}
	bool irrelevant(std::set<Line>::iterator it) {
		return hasPrev(it) && hasNext(it) && ((isMax && irrelevant(*std::prev(it), *it, *std::next(it)))
										  || (!isMax && irrelevant(*std::next(it), *it, *std::prev(it))));
	}

	std::set<Line>::iterator updateLeftBorder(std::set<Line>::iterator it) {
		if ((isMax && !hasPrev(it)) || (!isMax && !hasNext(it)))
			return it;

		double val = intersectX(*it, isMax ? *std::prev(it) : *std::next(it));
		Line buf(*it);
		it = hull.erase(it);
		buf.xLeft = val;
		it = hull.insert(it, buf);
		return it;
	}

public:
	ConvexHullDynamic(bool _isMax = 1) {
		isMax = true;
	}

	void addLine(coef_t a, coef_t b) {
		Line l3 = Line(a, b);
		auto it = hull.lower_bound(l3);

		if (it != hull.end() && areParallel(*it, l3)) {
			if ((isMax && it->b < b) || (!isMax && it->b > b))
				it = hull.erase(it);
			else
				return;
		}

		it = hull.insert(it, l3);
		if (irrelevant(it)) {
			hull.erase(it);
			return;
		}

		while (hasPrev(it) && irrelevant(std::prev(it)))
			hull.erase(std::prev(it));
		while (hasNext(it) && irrelevant(std::next(it)))
			hull.erase(std::next(it));

		it = updateLeftBorder(it);
		if (hasPrev(it))
			updateLeftBorder(std::prev(it));
		if (hasNext(it))
			updateLeftBorder(std::next(it));
	}
	
	val_t getBest(coord_t x) const {
		if (hull.size() == 0) {
			return -INF;
		}
		Line q;
		q.val = x;
		q.type = isMax ? Line::Type::maxQuery : Line::Type::minQuery;

		auto bestLine = hull.lower_bound(q);
		if (isMax)
			--bestLine;
		return bestLine->valueAt(x);
	}
};
//Convex Hull Dynamic long end

//O(V^2E) Dinic Flow
//Initialize : MaxFlow<# of vertices, Max Value> M;

template<int MX, ll INF> struct MaxFlow //by yutaka1999, have to define INF and MX (the Max number of vertices)
{
	struct edge
	{
		int to,cap,rev;
		edge(int to=0,int cap=0,int rev=0):to(to),cap(cap),rev(rev){}
	};
	vector <edge> vec[MX];
	int level[MX];
	int iter[MX];
	
	void addedge(int s,int t,int c) //adds an edge of cap c to the flow graph
	{
		int S=vec[s].size(),T=vec[t].size();
		vec[s].push_back(edge(t,c,T));
		vec[t].push_back(edge(s,0,S));
	}
	void bfs(int s)
	{
		memset(level,-1,sizeof(level));
		queue <int> que;
		level[s] = 0;
		que.push(s);
		while (!que.empty())
		{
			int v = que.front();que.pop();
			for(int i=0;i<vec[v].size();i++)
			{
				edge&e=vec[v][i];
				if (e.cap>0&&level[e.to]<0)
				{
					level[e.to]=level[v]+1;
					que.push(e.to);
				}
			}
		}
	}
	ll flow_dfs(int v,int t,ll f)
	{
		if (v==t) return f;
		for(int &i=iter[v];i<vec[v].size();i++)
		{
			edge &e=vec[v][i];
			if (e.cap>0&&level[v]<level[e.to])
			{
				ll d=flow_dfs(e.to,t,min(f,ll(e.cap)));
				if (d>0)
				{
					e.cap-=d;
					vec[e.to][e.rev].cap+=d;
					return d;
				}
			}
		}
		return 0;
	}
	ll maxflow(int s,int t) //finds max flow using dinic from s to t
	{
		ll flow = 0;
		while(1)
		{
			bfs(s);
			if (level[t]<0) return flow;
			memset(iter,0,sizeof(iter));
			while (1)
			{
				ll f=flow_dfs(s,t,INF);
				if(f==0) break;
				flow += f;
			}
		}
	}
};
//Dinic Flow end

//Hopkroft-Karp matching (MCBM, max-cardinality bipartite matching) start
//Read n1,n2 -> init() -> addEdge() -> maxMatching()
const int MAXN1 = 50000;
const int MAXN2 = 50000;
const int MAXM = 150000;

int n1, n2, edges, last[MAXN1], pre[MAXM], head[MAXM];
int matching[MAXN2], dist[MAXN1], Q[MAXN1];
bool used[MAXN1], vis[MAXN1];

void init(int _n1, int _n2) 
{
    n1 = _n1;
    n2 = _n2;
    edges = 0;
    fill(last, last + n1, -1);
}

void addEdge(int u, int v) 
{
    head[edges] = v;
    pre[edges] = last[u];
    last[u] = edges++;
}

void bfs() 
{
    fill(dist, dist + n1, -1);
    int sizeQ = 0;
    for (int u = 0; u < n1; ++u) {
        if (!used[u]) {
            Q[sizeQ++] = u;
            dist[u] = 0;
        }
    }
    for (int i = 0; i < sizeQ; i++) {
        int u1 = Q[i];
        for (int e = last[u1]; e >= 0; e = pre[e]) {
            int u2 = matching[head[e]];
            if (u2 >= 0 && dist[u2] < 0) {
                dist[u2] = dist[u1] + 1;
                Q[sizeQ++] = u2;
            }
        }
    }
}

bool dfs(int u1) 
{
    vis[u1] = true;
    for (int e = last[u1]; e >= 0; e = pre[e]) {
        int v = head[e];
        int u2 = matching[v];
        if (u2 < 0 || ((!vis[u2] && dist[u2] == dist[u1] + 1) && dfs(u2))) {
            matching[v] = u1;
            used[u1] = true;
            return true;
        }
    }
    return false;
}

int maxMatching() 
{
    fill(used, used + n1, false);
    fill(matching, matching + n2, -1);
    for (int res = 0;;) {
        bfs();
        fill(vis, vis + n1, false);
        int f = 0;
        for (int u = 0; u < n1; ++u)
            if (!used[u] && dfs(u))
                ++f;
        if (!f)
            return res;
        res += f;
    }
}
//Hopkroft-Karp matching end

//SCC (Strongly connected components) start
//init(n) -> read input -> tarjan() -> sccidx[]
struct SCC
{
	const int INF2 = int(1e9);
	vector<vector<int> > vec;
	int index;
	vector<int> idx;
	vector<int> lowlink;
	vector<bool> onstack;
	stack<int> s;
	vector<int> sccidx;
	int scccnt;
	vi topo;
	
	//lower sccidx means appear later
	void init(int n)
	{
		idx.assign(n,-1);
		index = 0;
		onstack.assign(n,0);
		lowlink.assign(n,INF2);
		while(!s.empty()) s.pop();
		sccidx.assign(n,-1);
		scccnt = 0;
		vec.clear();
		topo.clear();
		vec.resize(n);
	}
	
	void addedge(int u, int v) //u -> v
	{
		vec[u].pb(v);
	}
	
	void connect(int u)
	{
		idx[u] = index;
		lowlink[u] = index;
		index++;
		s.push(u);
		onstack[u] = true;
		for(int i = 0; i < vec[u].size(); i++)
		{
			int v = vec[u][i];
			if(idx[v] == -1)
			{
				connect(v);
				lowlink[u] = min(lowlink[u], lowlink[v]);
			}
			else if(onstack[v])
			{
				lowlink[u] = min(lowlink[u], idx[v]);
			}
		}
		if(lowlink[u] == idx[u])
		{
			while(1)
			{
				int v = s.top();
				s.pop();
				onstack[v] = false;
				sccidx[v] = scccnt;
				if(v == u) break;
			}
			scccnt++;
		}
	}
	
	void tarjan()
	{
		for(int i = 0; i < vec.size(); i++)
		{
			if(idx[i] == -1)
			{
				connect(i);
			}
		}
	}
	
	void toposort() //if graph is a DAG and i just want to toposort
	{
		tarjan();
		int n = vec.size();
		topo.resize(n);
		vector<ii> tmp;
		for(int i = 0; i < n; i++)
		{
			tmp.pb(ii(sccidx[i],i));
		}
		sort(tmp.begin(),tmp.end());
		reverse(tmp.begin(),tmp.end());
		for(int i = 0; i < n; i++)
		{
			topo[i]=tmp[i].S;
			if(i>0) assert(tmp[i].F!=tmp[i-1].F);
		}
	}
};
//SCC end

//Binary converter start
string BinToString(ll x)
{
	string res;
	for(int i=8;i>=0;i--){
		if((1LL<<i)&x) res+='1';
		else res+='0';
	}
	return res;
}
//Binary converter end

//Grid movement (4-direction) start
int dx[4]={-1,1,0,0};
int dy[4]={0,0,-1,1};
bool oob(int x, int y){
	return x<0 || y<0 || x>=n || y>=m;
}

forn(k,0,4){
	int x1=x+dx[k], y1=y+dy[k];
	if(oob(x1,y1)) continue;
	if(vst[x1][y1]) continue;
	
	
}
//Grid movement (4-direction) end

//Grid movement (8-direction) start
int dx[8]={-1,0,1,-1,1,-1,0,1};
int dy[8]={-1,-1,-1,0,0,1,1,1};
//Grid movement (8-direction) end
