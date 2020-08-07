#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;
 
#define fi first
#define se second
#define mp make_pair
#define pb push_back
#define fbo find_by_order
#define ook order_of_key
 
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef long double ld; 
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> pbds;

int cc;

struct DSU
{
	int S;
	
	vector<pair<ii,int> > updates;
	struct node
	{
		int p; ll sum;
	};
	vector<node> dsu;
	
	DSU(int n)
	{
		S = n;
		for(int i = 0; i < n; i++)
		{
			node tmp;
			tmp.p = i; tmp.sum = 1;
			dsu.pb(tmp);
		}
	}
	
	void reset(int n)
	{
		dsu.clear();
		S = n;
		for(int i = 0; i < n; i++)
		{
			node tmp;
			tmp.p = i; tmp.sum = 1;
			dsu.pb(tmp);
		}
	}
	
	int rt(int u)
	{
		if(dsu[u].p == u) return u;
		return rt(dsu[u].p);
	}
	
	void merge(int u, int v)
	{
		u = rt(u); v = rt(v);
		if(u == v) 
		{
			updates.pb({{-1,-1}, -1});
			return ;
		}
		if(dsu[u].sum < dsu[v].sum) swap(u, v);
		updates.pb({{v,dsu[v].p}, dsu[v].sum}); //dsu[v].p should have been dsu[v].p
		dsu[v].p = u;
		dsu[u].sum += dsu[v].sum;
		cc--;
	}
	
	bool sameset(int u, int v)
	{
		if(rt(u) == rt(v)) return true;
		return false;
	}
	
	ll getstat(int u)
	{
		return dsu[rt(u)].sum;
	}
	
	void undo()
	{
		assert(!updates.empty());
		int u = updates.back().fi.fi; int v = updates.back().fi.se; int w = updates.back().se; updates.pop_back();
		if(u==-1) return ;
		int curp = dsu[u].p;
		dsu[curp].sum -= w;
		dsu[u].p = v;
		dsu[u].sum = w;
		cc++;
	}
};

int main()
{
	//ios_base::sync_with_stdio(0); cin.tie(0);
	int n,q; scanf("%d %d", &n, &q);
	cc=q;
	DSU dsu(q);
	for(int i=0;i<n;i++)
	{
		//string s; //cin>>s;
		char f[11];
		scanf("%s", f);
		if(f[1]=='O')
		{
			dsu.undo();
		}
		else
		{
			int u,v; scanf("%d %d",&u,&v); u--; v--;
			dsu.merge(u,v);
		}
		printf("%d\n", cc);
	}
}
