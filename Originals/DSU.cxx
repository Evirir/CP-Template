struct DSU{
	int S;
	
	struct node{
		int p;
		ll sum;
	};
	vector<node> dsu;
	
	DSU(int n){
		S = n;
		dsu.resize(S);
		forn(i,0,n){
			dsu[i].p=i; dsu[i].r=0; dsu[i].sum=0;
		}
	}
	
	int rt(int u){
		if(dsu[u].p == u) return u;
		dsu[u].p = rt(dsu[u].p);
		return dsu[u].p;
	}
	void merge(int u, int v){
		u = rt(u); v = rt(v);
		if(u == v) return;
		if(rand()&1) swap(u,v);
		dsu[v].p = u;
		dsu[u].sum += dsu[v].sum;
	}
	bool sameset(int u, int v){
		if(rt(u) == rt(v)) return true;
		return false;
	}
	ll getstat(int u){
		return dsu[rt(u)].sum;
	}
};
