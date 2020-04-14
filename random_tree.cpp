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

void generate_tree_adj(int n){
	cout<<n<<'\n';
	fore(i,2,n){
		cout<<i<<" "<<uniform_int_distribution<int>(1,i-1)(rng)<<'\n';
	}
}

void generate_tree_prt(int n){
	cout<<n<<'\n';
	fore(i,2,n){
		cout<<uniform_int_distribution<int>(1,i-1)(rng)<<" ";
	}
	cout<<'\n';
}

int main()
{
	ios_base::sync_with_stdio(0); cin.tie(0);
	
	int n; cin>>n;
	generate_tree_prt(n);
	
	return 0;
}
