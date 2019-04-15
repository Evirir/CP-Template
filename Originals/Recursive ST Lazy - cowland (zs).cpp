#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;
 
#define fi first
#define se second
#define mp make_pair
#define pb push_back
 
typedef long long ll;
typedef pair<int,int> ii;
typedef vector<int> vi;
typedef unsigned long long ull;
typedef long double ld; 
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> pbds;

class LazySegmentTree {
private:
	int size_;
	vector<long long> v, lazy;
	
	void update(int a, int b, long long x, int k, int l, int r) {
		push(k, l, r);
		if (r <= a || b <= l) return;
		if (a <= l && r <= b) {
			lazy[k] = x;
			push(k, l, r);
		}
		else {
			update(a, b, x, k * 2, l, (l + r) >> 1);
			update(a, b, x, k * 2 + 1, (l + r) >> 1, r);
			v[k] = merge(v[k * 2], v[k * 2 + 1]);
		}
	}
	
	long long query(int a, int b, int k, int l, int r) {
		push(k, l, r);
		if (r <= a || b <= l) return 0;
		if (a <= l && r <= b) return v[k];
		long long lc = query(a, b, k * 2, l, (l + r) >> 1);
		long long rc = query(a, b, k * 2 + 1, (l + r) >> 1, r);
		return merge(lc, rc);
	}
 
public:
	LazySegmentTree() : v(vector<long long>()), lazy(vector<long long>()) {};
	LazySegmentTree(int n) {
		for (size_ = 1; size_ < n;) size_ <<= 1;
		v.resize(size_ * 2);
		lazy.resize(size_ * 2);
	}
	inline void push(int k, int l, int r) {
		if (lazy[k] != 0) {
			v[k] ^= lazy[k];
			if (r - l > 1) {
				lazy[k * 2] ^= lazy[k];
				lazy[k * 2 + 1] ^= lazy[k];
			}
			lazy[k] = 0;
		}
	}
	inline long long merge(long long x, long long y) {
		return x^y;
	}
	inline void update(int l, int r, long long x) {
		update(l, r, x, 1, 0, size_);
	}
	inline long long query(int l, int r) {
		return query(l, r, 1, 0, size_);
	}
};

const int LG = 18;

ll a[222222];
ll xr[222222];
vi adj[222222];
int s[222222];
int e[222222];
int timer=-1;
int h[222222];			//depth
int st[LG+1][222222];	//parent

void dfs(int u, int p)
{
	xr[u]^=a[u];
	s[u]=++timer;
	st[0][u]=p;
	for(int v:adj[u])
	{
		if(v==p) continue;
		h[v]=h[u]+1;
		xr[v]^=xr[u];
		dfs(v,u);
	}
	e[u]=timer;
}

int lca(int u, int v)
{
	if(h[u]>h[v]) swap(u,v);
	for(int i=LG-1;i>=0;i--)
	{
		if(st[i][v]!=-1&&h[st[i][v]]>=h[u])
		{
			v=st[i][v];
		}
	}
	if(u==v) return u;
	for(int i=LG-1;i>=0;i--)
	{
		if(st[i][v]!=-1&&st[i][v]!=st[i][u])
		{
			v=st[i][v];u=st[i][u];
		}
	}
	return st[0][u];
}

int main()
{
	ios_base::sync_with_stdio(0); cin.tie(0);
	//freopen("cowland.in","r",stdin); freopen("cowland.out","w",stdout);
	int n,q; cin>>n>>q;
	for(int i=0;i<n;i++)
	{
		cin>>a[i];
	}
	for(int i=0;i<n-1;i++)
	{
		int u,v; cin>>u>>v; u--; v--;
		adj[u].pb(v); adj[v].pb(u);
	}
	dfs(0,-1);
	for(int i=1;i<LG;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(st[i-1][j]==-1) st[i][j]=-1;
			else st[i][j]=st[i-1][st[i-1][j]];
		}
	}	
	LazySegmentTree st(n+11);
	for(int i=0;i<n;i++) st.update(s[i],s[i]+1,xr[i]);
	for(int i=0;i<q;i++)
	{
		int t; cin>>t;
		if(t==1)
		{
			int u,v; cin>>u>>v; u--; 
			st.update(s[u],e[u]+1,v^a[u]);
			a[u]=v;
		}
		else
		{
			int u,v; cin>>u>>v; u--; v--;
			ll A = (st.query(s[u],s[u]+1));
			ll B = (st.query(s[v],s[v]+1));
			cout<<(A^B^a[lca(u,v)])<<'\n';
		}
	}
}
