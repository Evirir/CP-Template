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
#define INF 0x3f3f3f3f
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

struct NumberTheory{
	vector<ll> primes;
	vector<bool> prime;
	
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
}
