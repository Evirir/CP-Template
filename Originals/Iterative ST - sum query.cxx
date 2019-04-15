#include <bits/stdc++.h>
using namespace std;

#define debug(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)
#define time__() for(long blockTime=NULL;(blockTime==NULL?(blockTime=clock())!=NULL:false); debug("Time:%.4fs\n",(double)(clock()-blockTime)/CLOCKS_PER_SEC))
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=a;i<b;i++)
#define fore(i,a,b) for(int i=a;i<=b;i++)
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define PI 3.14159265358979323846264338327
#define INF 2e9
#define MOD 998244353
#define MAXN 111111
typedef long long ll;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef unsigned long long ull;

int n,q;
int t[2*MAXN];

void build(){
	for(int i=n-1;i>0;i--) 	t[i]=t[2*i]+t[2*i+1];	
}

void update(int p,int val){
	p--;
	for(t[p+=n]=val;p>1;p/=2)	t[p/2]=t[p]+t[p^1];
}

int query(int l,int r){
	l--;
	int sum=0;
	for(l+=n,r+=n;l<r;l/=2,r/=2){
		if(l&1)	sum+=t[l++];
		if(r&1)	sum+=t[--r];
	}
	return sum;
}

int main()
{
	//ios_base::sync_with_stdio(0); cin.tie(0);
	
	cin>>n>>q;
	forn(i,0,n)		cin>>t[n+i];
	build();
	forn(it,0,q){
		char c;	cin>>c;
		if(c=='q'){
			int l,r;	cin>>l>>r;
			cout<<query(l,r)<<'\n';
		}
		else{
			int p,val;	cin>>p>>val;
			update(p,val);
			forn(i,0,n)	cout<<t[n+i]<<" ";
			cout<<'\n';
		}
	}
	
	return 0;
}
