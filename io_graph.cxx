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
#define F first
#define S second
#define PI 3.14159265358979323846264338327
#define INF 2e14
#define MOD 998244353
#define pqueue priority_queue
#define fbo find_by_order
#define ook order_of_key
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef unsigned long long ull;
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> pbds;

#define MAXN 100005

struct DSU{
	struct node{
		int p,r;
	};
	vector<node> dsu;
	
	DSU(int n){
		forn(i,0,n){
			node tmp;
			tmp.p=i; tmp.r=0;
			dsu.pb(tmp);
		}
	}
	
	int rt(int u){
		if(dsu[u].p!=u)	dsu[u].p=rt(dsu[u].p);
		return dsu[u].p;
	}
	
	void merge(int u,int v){
		int ur=rt(u),vr=rt(v);
		if(ur==vr) return;
		if(dsu[ur].r<dsu[vr].r)	dsu[ur].p=vr;
		else if(dsu[ur].r>dsu[vr].r) dsu[vr].p=ur;
		else{
			dsu[ur].r++;
			dsu[vr].p=ur;
		}
	}
	
	bool sameset(int u,int v){
		if(dsu[u].p==dsu[v].p)	return true;
		return false;
	}
	
};

struct Graph{
	struct edge{
		int v;	ll wgt;
	};
	int n;
	vector<vector<edge> > adj;
	Graph(int n1){
		adj.resize(n1);
		n=n1;
	}
	void addedge(int u,int v,ll wgt){	//undirected graph
		edge tmp;
		tmp.wgt=wgt;
		tmp.v=v;	adj[u].pb(tmp);
		tmp.v=u;	adj[v].pb(tmp);
	}
	
	bool vst[MAXN];
	ll d[MAXN];
	vi par;
	
	void dfs(int u){
		if(vst[u])	return;
		vst[u]=true;
		//process node here
		forn(i,0,adj[u].size())	dfs(adj[u][i].v);
	}
	void bfs(int u){
		if(vst[u])	return;
		vst[u]=true;
		queue<int> Q;	Q.push(u);
		while(!Q.empty()){
			int v=Q.front();	Q.pop();
			//process node here
			forn(i,0,adj[v].size()){
				int x=adj[v][i].v;
				if(!vst[x])	Q.push(x);
			}			
		}
	}
	
	void dijkstra(int s){
		forn(i,0,n){
			d[i]=INF;
			par[i]=-1;
		}
		d[s]=0;
		pqueue<ii,vii,greater<ii> > q;
		q.push(mp(0,s));
		while(!q.empty()){
			int u=q.top().S;
			q.pop();
			forn(i,0,adj[u].size()){
				int v=adj[u][i].v,w=adj[u][i].wgt;
				if(d[v]<d[u]+w){
					d[v]=d[u]+w;
					q.push(mp(d[v],v));
					par[v]=u;
				}
			}
		}
	}
	
	vector<pair<ii,ll> > mst;
	vector<pair<ll,ii> > edges;
	
	void kruskal(){
		DSU dsu(n);
		forn(i,0,n){
			forn(j,0,adj[i].size()){
				ll u=i, v=adj[i][j].v, w=adj[i][j].wgt;
				edges.pb(mp(w,mp(u,v)));
			}
		}
		sort(edges.begin(),edges.end());
		ll cnt=0;
		
		forn(i,0,edges.size()){
			ll u=edges[i].S.F, v=edges[i].S.S, w=edges[i].F;
			if(dsu.sameset(u,v))	continue;
			dsu.merge(u,v);
			cnt++;
			mst.pb(mp(mp(u,v),w));
			if(cnt==n-1)	break;
		}
	}
};

int main()
{
	ios_base::sync_with_stdio(0); cin.tie(0);
	
	
	
	return 0;
}
