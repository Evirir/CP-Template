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
		ll lc = query(s, e, k*2, l, (l+r)>>1);
		ll rc = query(s, e, k*2+1, ((l+r)>>1)+1, r);
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
				lazy[k*2].sum+=lazy[k].sum;
				lazy[k*2+1].sum+=lazy[k].sum;
				lazy[k*2].mn+=lazy[k].mn;
				lazy[k*2+1].mn+=lazy[k].mn;
				lazy[k*2].mx+=lazy[k].mx;
				lazy[k*2+1].mx+=lazy[k].mx;
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
		if(p < l || p > r) return;
		if(l == r){
			v[k]+=val;	//modification
			return;
		}
		int mid = (l+r)>>1;
		update(p, val, k*2, l, mid);
		update(p, val, k*2+1, mid+1, r);
		v[k] = merge(v[k*2], v[k*2+1]);
	}
	
	ll query(int s, int e, int k, int l, int r)
	{
		if(e < l || s > r) return 0; //dummy value
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
		if(p < l || p > r) return;
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
		if(e < l || s > r) return Node(0,INF,-1); //dummy value
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
	inline void update(int p, Node val){
		update(p, val, 1, 0, size_-1);
	}
	inline Node query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Point recursive with struct ST end

//Point iterative ST start
struct IterSegmentTree{
	void build(){
		for(int i=n-1; i>0; i--) t[i]=t[i<<1]+t[i<<1|1];
	}
	
	void modify(int p, int val){
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

//Start FenwickRange
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
    void update(int l, int r, ll val) //[l,r] + val
    {    
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
    ll query(ll l, ll r)
    {
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
};
//End FenwickRange

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
	//ll get(int u){ return dsu[rt(u)].sum; }
	//void set(int u, ll val){ dsu[rt(u)].sum = val; }
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

void dijkstra(int src)
{
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
	forn(i,0,n) forn(j,0,n) dist[i][j] = (adj[i][j]==0 ? INF : adj[i][j]);
	forn(k,0,n) forn(i,0,n) forn(j,0,n)
		dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}
//Floyd end

//HLD/Euler path start
#define LG 19

vi adj[MAXN];
int in[MAXN],out[MAXN];
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
//Euler path end

ll Query(int u,int v){
	ll ans=0;
	while(top[u]!=top[v]){
		if(dep[top[u]]<dep[top[v]]) swap(u,v);
		ans+=st.query(in[top[u]],in[u]);
		u=prt[top[u]];
	}
	
	if(dep[u]<dep[v]) swap(u,v);
	return ans+st.query(in[v],in[u]);
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
#define LG 25

ll spt[MAXN][LG+1];
int lg[MAXN+1];

struct SparseTable
{	
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
//Sparse Table end

//LCA start
#define LG 20

int dep[MAXN],prt[MAXN][LG];
mset(prt,-1);

void dfs(int u, int p)
{
	prt[u][0]=p;
	forn(j,1,LG){
		if(prt[u][j-1]!=-1) prt[u][j]=prt[prt[u][j-1]][j-1];
	}
	for(int v:adj[u])
	{
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs(v,u);
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

//Combi/Maths start
struct Maths
{
	vector<ll> fact,ifact,inv,pow2;
	ll add(ll a,ll b)
	{
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
	ll choose(ll a, ll b)
	{
		if(a<b) return 0;
		if(b==0) return 1;
		if(a==b) return 1;
		return mult(fact[a],mult(ifact[b],ifact[a-b]));
	}
	ll inverse(ll a)
	{
		return pw(a,MOD-2);
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
		    ifact[i] = mult(ifact[i + 1], i + 1);
		}
		for(int i=1;i<=_n;i++){
		    inv[i] = mult(fact[i-1],ifact[i]);
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
};
//Combi/Maths end

//NT start
struct NumberTheory
{
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

	//ll op;
	void getDiv(vector<ll>& div, vector<ii>& pf, ll n, int i)
	{
		//op++;
		ll x, k;
		if(i >= pf.size()) return ;
		x = n;
		for(k = 0; k <= pf[i].S; k++)
		{
			if(i==int(pf.size())-1) div.pb(x);
			getDiv(div, pf, x, i + 1);
			x *= pf[i].F;
		}
	}
};
//End NT
