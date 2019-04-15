vii edge[MAXN];
ll dist[MAXN];

int n;

void dijkstra(int src){
	pqueue<ii,vii,greater<ii>> q;
	forn(i,0,n)	dist[i]=INF;
	dist[src]=0;
	q.push({0,src});
	
	while(!q.empty()){
		int u=q.top().S; q.pop();
		for(auto tmp: adj[u]){
			int v=tmp.S; ll w=tmp.F;
			if(dist[v]>dist[u]+w){
				dist[v]=dist[u]+w;
				q.push({d[v],v});
			}
		}
	}
}
