#include <bits/stdc++.h>
using namespace std;

#define debug(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)
#define time__() for(long blockTime=NULL;(blockTime==NULL?(blockTime=clock())!=NULL:false); debug("Time:%.4fs\n",(double)(clock()-blockTime)/CLOCKS_PER_SEC))
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=a;i<b;i++)
#define fore(i,a,b) for(int i=a;i<=b;i++)
#define pqueue priority_queue
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define PI 3.14159265358979323846264338327
#define INF 0x3f3f3f3f
#define MOD 998244353
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef unsigned long long ull;

struct DSU{
	struct node{
		int p,r;
	};
	vector<node> dsu;
	
	DSU(int n){
		forn(i,0,n){
			node tmp;
			tmp.p=i;	tmp.r=0;
			dsu.pb(tmp);
		}
	}
	
	int rt(int u){
		if(dsu[u].p!=u)	dsu[u].p=rt(dsu[u].p);
		return dsu[u].p;
	}
	
	void merge(int u,int v){
		int ur=rt(u),vr=rt(v);
		if(ur==vr)	return;
		if(dsu[ur].r<dsu[vr].r)	dsu[ur].p=vr;
		else if(dsu[ur].r>dsu[vr].r)	dsu[vr].p=ur;
		else{
			dsu[ur].r++;
			dsu[vr].p=ur;
		}
	}
	
	bool sameset(int u,int v){
		if(rt(u)==rt(v))	return true;
		return false;
	}
};

int V,E;
vector<pair<ll,ii> > edges;
vector<pair<ii,ll> > mst;
ll tgwt=0;

void kruskal(){
	DSU dsu(V);
	forn(i,0,E){
		int u,v,w;	cin>>u>>v>>w;
		edges.pb(mp(w,mp(u,v)));
	}
	
	sort(edges.begin(),edges.end());
	ll cnt=0;
	
	forn(i,0,E){
		int u=edges[i].S.F,v=edges[i].S.S,w=edges[i].F;
		if(dsu.sameset(u,v))	continue;
		dsu.merge(u,v);
		mst.pb(mp(mp(u,v),w));
		cnt++; tgwt+=w;
		if(cnt>=V-1)	break;
	}	
}

int main()
{
	ios_base::sync_with_stdio(0); cin.tie(0);
	
	cin>>V>>E;
	kruskal();
	cout<<"\nMST:\n";
	forn(i,0,mst.size()){
		cout<<mst[i].F.F<<" "<<mst[i].F.S<<" ("<<mst[i].S<<")\n";
	}
	cout<<"Total weight: "<<tgwt<<'\n';
	
	return 0;
}
