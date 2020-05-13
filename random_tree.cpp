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

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

vector<int> perm;

void generate_tree_adj(int n){
	cout<<n<<'\n';
	forn(i,1,n){
		cout<<perm[i]+1<<" "<<perm[uniform_int_distribution<int>(0,i-1)(rng)]+1<<'\n';
	}
}

void generate_tree_prt(int n){
	int prt[n]; mset(prt,-1);
	cout<<n<<'\n';
	forn(i,1,n){
		prt[perm[i]]=perm[uniform_int_distribution<int>(0,i-1)(rng)];
	}
	forn(i,1,n){
		cout<<prt[i]+1<<" ";
	}
	cout<<'\n';
}

void prt_to_adj(){
	int n; cin>>n;
	int a[n];
	forn(i,1,n) cin>>a[i];
	
	cout<<n<<'\n';
	forn(i,1,n) cout<<i+1<<" "<<a[i]+1<<'\n';
}

int main()
{
	ios_base::sync_with_stdio(0); cin.tie(0);
	
	int n=20;
	
	perm.resize(n);
	forn(i,0,perm.size()) perm[i]=i;
	shuffle(perm.begin()+1,perm.end(),rng);
	
	generate_tree_prt(n); cout<<flush;
	prt_to_adj();
	
	return 0;
}
