#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

#define watch(x) cout<<(#x)<<"="<<(x)<<'\n'
#define mset(d,val) memset(d,val,sizeof(d))
#define setp(x) cout<<fixed<<setprecision(x)
#define forn(i,a,b) for(int i=(a);i<(b);i++)
#define fore(i,a,b) for(int i=(a);i<=(b);i++)
#define pb push_back
#define F first
#define S second
#define pqueue priority_queue
#define fbo find_by_order
#define ook order_of_key
typedef long long ll;
typedef pair<int,int> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
typedef long double ld;
template<typename T>
using pbds = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
void amin(ll &a, ll b){ a=min(a,b); }
void amax(ll &a, ll b){ a=max(a,b); }
void YES(){cout<<"YES\n";} void NO(){cout<<"NO\n";}
void SD(int t=0){ cout<<"PASSED "<<t<<endl; }
const ll INF = ll(1e18);
const int MOD = 998244353;

const bool DEBUG = 0;
const int MAXN = 100005;

//Lazy Recursive ST start
class LazySegmentTree{
private:
	int size_;
	vector<ll> v,lazy;
	
	void update(int s, int e, ll val, int k, int l, int r){
		push(k, l, r);
		if(r < s || e < l) return;
		if(s <= l && r <= e){
			lazy[k] = val;
			push(k, l, r);
		}
		else{
			update(s, e, val, k*2, l, (l+r)>>1);
			update(s, e, val, k*2+1, ((l+r)>>1)+1, r);
			v[k] = merge(v[k*2], v[k*2+1]);
		}
	}
	
	ll query(int s, int e, int k, int l, int r){
		push(k, l, r);
		if(r < s || e < l) return 0; //dummy value
		if(s <= l && r <= e) return v[k];
		ll lc = query(s, e, k*2, l, (l+r)>>1);
		ll rc = query(s, e, k*2+1, ((l+r)>>1)+1, r);
		return merge(lc, rc);
	}
 
public:
	LazySegmentTree(): v(vector<ll>()), lazy(vector<ll>()) {}
	LazySegmentTree(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
		lazy.resize(size_*4);
	}
	void reset(){
		v.assign(size_*4,0);
		lazy.assign(size_*4,0);
	}
	inline void push(int k, int l, int r){
		if(lazy[k]!=0){
			v[k]+=(r-l+1)*lazy[k];	//remember to consider the range!
			if(l!=r){
				lazy[k*2]+=lazy[k];
				lazy[k*2+1]+=lazy[k];
			}
			lazy[k]=0;
		}
	}
	inline ll merge(ll x, ll y){
		return x+y;
	}
	inline void update(int l, int r, ll val){
		update(l, r, val, 1, 0, size_-1);
	}
	inline ll query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Lazy recursive ST end

//Lazy recursive ST with struct start
struct Node{
	ll sum,mn,mx;
	Node(){sum=mn=mx=0;}
	Node(ll _s,ll _mn,ll _mx){ sum=_s; mn=_mn; mx=_mx; }
};

class LazySegmentTreeNode{
private:	
	int size_;
	vector<Node> v,lazy;
	
	void update(int s, int e, const Node &val, int k, int l, int r){
		push(k, l, r);
		if(r < s || e < l) return;
		if(s <= l && r <= e){
			lazy[k] = val;
			push(k, l, r);
		}
		else{
			update(s, e, val, k*2, l, (l+r)>>1);
			update(s, e, val, k*2+1, ((l+r)>>1)+1, r);
			v[k] = merge(v[k*2], v[k*2+1]);
		}
	}
	
	Node query(int s, int e, int k, int l, int r){
		push(k, l, r);
		if(r < s || e < l) return Node(0,INF,-1); //dummy value
		if(s <= l && r <= e) return v[k];
		Node lc = query(s, e, k*2, l, (l+r)>>1);
		Node rc = query(s, e, k*2+1, ((l+r)>>1)+1, r);
		return merge(lc, rc);
	}
 
public:
	LazySegmentTreeNode(): v(vector<Node>()), lazy(vector<Node>()) {}
	LazySegmentTreeNode(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
		lazy.resize(size_*4);
	}
	//void reset(){}
	inline void push(int k, int l, int r){
		if(lazy[k]!=Node(0,0,0)){
			v[k].sum+=(r-l+1)*lazy[k].sum;	//remember to consider the range!
			v[k].mn+=lazy[k].mn;
			v[k].mx+=lazy[k].mx;
			if(l!=r){
				lazy[k*2].sum+=lazy[k].sum;	lazy[k*2+1].sum+=lazy[k].sum;
				lazy[k*2].mn+=lazy[k].mn;	lazy[k*2+1].mn+=lazy[k].mn;
				lazy[k*2].mx+=lazy[k].mx;	lazy[k*2+1].mx+=lazy[k].mx;
			}
			lazy[k]=Node(0,0,0);
		}
	}
	inline Node merge(const Node &x, const Node &y){
		Node tmp;
		tmp.sum = x.sum + y.sum;
		tmp.mn = min(x.mn, y.mn);
		tmp.mx = max(x.mx, y.mx);
		return tmp;
	}
	inline void update(int l, int r, const Node &val){
		update(l, r, val, 1, 0, size_-1);
	}
	inline Node query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Lazy recursive ST with struct end

//Point recursive ST start
class PointSegmentTree{
private:
	int size_;
	vector<ll> v;
	
	void update(int p, ll val, int k, int l, int r)
	{
		if(p < l || r < p) return;
		if(l == r){
			v[k]=val;	//modification
			return;
		}
		int mid = (l+r)>>1;
		update(p, val, k*2, l, mid);
		update(p, val, k*2+1, mid+1, r);
		v[k] = merge(v[k*2], v[k*2+1]);
	}
	
	ll query(int s, int e, int k, int l, int r)
	{
		if(e < l || r < s) return 0; //dummy value
		if(s <= l && r <= e) return v[k];
		int mid = (l+r)>>1;
		ll lc = query(s, e, k*2, l, mid);
		ll rc = query(s, e, k*2+1, mid+1, r);
		return merge(lc, rc);
	}
	
public:
	PointSegmentTree(): v(vector<ll>()) {}
	PointSegmentTree(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
	}
	//void reset(){}
	inline ll merge(ll x, ll y){
		return x+y;
	}
	inline void update(int p, ll val){
		update(p, val, 1, 0, size_-1);
	}
	inline ll query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Point recursive ST end

//Point recursive ST with struct start
struct Node{
	ll sum,mn,mx;
	Node(){sum=mn=mx=0;}
	Node(ll _s,ll _mn,ll _mx){ sum=_s; mn=_mn; mx=_mx; }
};

class PointSegmentTreeNode{
private:
	int size_;
	vector<Node> v;
	
	void update(int p, const Node &val, int k, int l, int r)
	{
		if(p < l || r < p) return;
		if(l == r){
			v[k].sum += val.sum; //modifications
			v[k].mx = val.mx;
			v[k].mn = val.mn;
			return;
		}
		int mid = (l+r)>>1;
		update(p, val, k*2, l, mid);
		update(p, val, k*2+1, mid+1, r);
		v[k] = merge(v[k*2], v[k*2+1]);
	}
	
	Node query(int s, int e, int k, int l, int r)
	{
		if(e < l || r < s) return Node(0,INF,-1); //dummy value
		if(s <= l && r <= e) return v[k];
		int mid = (l+r)>>1;
		Node lc = query(s, e, k*2, l, mid);
		Node rc = query(s, e, k*2+1, mid+1, r);
		return merge(lc, rc);
	}
	
public:
	PointSegmentTreeNode(): v(vector<Node>()) {}
	PointSegmentTreeNode(int n){
		for(size_=1;size_<n;) size_<<=1;
		v.resize(size_*4);
	}
	//void reset(){}
	inline Node merge(const Node &x, const Node &y){
		Node tmp;
		tmp.sum = x.sum + y.sum;
		tmp.mn = min(x.mn, y.mn);
		tmp.mx = max(x.mx, y.mx);
		return tmp;
	}
	inline void update(int p, ll _sm=0, ll _mn=INF, ll _mx=0){
		update(p, Node(_sm,_mn,_mx), 1, 0, size_-1);
	}
	inline Node query(int l, int r){
		return query(l, r, 1, 0, size_-1);
	}
};
//Point recursive with struct ST end

//Point iterative ST start
struct IterSegmentTree{
	vector<ll> t;
	
	IterSegmentTree: t(vector<ll>()) {}
	IterSegmentTree(int _n){
		t.resize(_n*2);
	}
	
	void build(){
		for(int i=n-1; i>0; i--) t[i]=t[i<<1]+t[i<<1|1];
	}
	
	void update(int p, int val){
		for(t[p+=n]=value; p>1; p>>=1) t[p>>1]=t[p]+t[p^1];
	}
	
	int query(int l, int r){
		int res=0;
		for(l+=n,r+=n; l<r; l>>=1,r>>=1){
			if(l&1) res+=t[l++];
			if(r&1) res+=t[--r];
		}
		return res;
	}
};

forn(i,0,n) cin>>t[n+i];
//Point iterative ST end

//Fenwick Tree (FenwickPoint) start
struct FenwickPoint{
	vector<ll> fw;
	int siz;
	FenwickPoint(): fw(vector<ll>()), siz(0) {}
	FenwickPoint(int N)
	{
		fw.assign(N+1,0);
		siz = N+1;
	}
	void reset(int N)
	{
		fw.assign(N+1,0);
		siz = N+1;
	}
	void add(int p, ll val)
	{
		for(p++; p<siz; p+=(p&(-p)))
		{
			fw[p]+=val;
		}
	}
	ll sum(int p)
	{
		ll res=0;
		for(; p; p-=(p&(-p)))
		{
			res+=fw[p];
		}
		return res;
	}
	ll query(int l, int r)
	{
		l++; r++;
		if(r<l) return 0;
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
	void modify(int p, ll val)
	{
		add(p, val-query(p,p));
	}
};
//Fenwick Tree (FenwickPoint) end

//FenwickRange start
struct FenwickRange
{
	vector<ll> fw,fw2;
	int siz;
	FenwickRange(): fw(vector<ll>()), fw2(vector<ll>()), siz(0) {}
	FenwickRange(int N)
	{
		fw.assign(N+1,0);
		fw2.assign(N+1,0);
		siz = N+1;
	}
	void reset(int N)
	{
		fw.assign(N+1,0);
		fw2.assign(N+1,0);
		siz = N+1;
	}
	void add(int l, int r, ll val) //[l,r] + val
	{   
		l++; r++;
		for(int tl=l; tl<siz; tl+=(tl&(-tl)))
		{
			fw[tl]+=val, fw2[tl]-=val*ll(l-1);
		}
		for(int tr=r+1; tr<siz; tr+=(tr&(-tr)))
		{
			fw[tr]-=val, fw2[tr]+=val*ll(r);
		}
	}
	ll sum(int r) //[1,r]
	{                         
		ll res=0;
		for(int tr=r; tr; tr-=(tr&(-tr)))
		{
			res+=fw[tr]*ll(r)+fw2[tr];
		}
		return res;
	}
	ll query(int l, int r)
	{
		l++; r++;
		if(r<l) return 0;
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
	void modify(int p, ll val)
	{
		add(p,p,val-query(p,p));
	}
};
//FenwickRange end

//Segment Tree Beats start (by yaketake08/tjake)
//https://tjkendev.github.io/procon-library/cpp/range_query/segment_tree_beats_2.html
//All intervals are [L,R)

#define N MAXN
class SegmentTree {
	const ll inf = 1e18;
	int n, n0;
	ll max_v[4*N], smax_v[4*N], max_c[4*N];
	ll min_v[4*N], smin_v[4*N], min_c[4*N];
	ll sum[4*N];
	ll len[4*N], ladd[4*N], lval[4*N];

	void update_node_max(int k, ll x) {
		sum[k] += (x - max_v[k]) * max_c[k];

		if(max_v[k] == min_v[k]) {
			max_v[k] = min_v[k] = x;
		} else if(max_v[k] == smin_v[k]) {
			max_v[k] = smin_v[k] = x;
		} else {
			max_v[k] = x;
		}

		if(lval[k] != inf && x < lval[k]) {
			lval[k] = x;
		}
	}
	void update_node_min(int k, ll x) {
		sum[k] += (x - min_v[k]) * min_c[k];

		if(max_v[k] == min_v[k]) {
			max_v[k] = min_v[k] = x;
		} else if(smax_v[k] == min_v[k]) {
			min_v[k] = smax_v[k] = x;
		} else {
			min_v[k] = x;
		}

		if(lval[k] != inf && lval[k] < x) {
			lval[k] = x;
		}
	}

	void push(int k) {
		if(n0-1 <= k) return;

		if(lval[k] != inf) {
			updateall(2*k+1, lval[k]);
			updateall(2*k+2, lval[k]);
			lval[k] = inf;
			return;
		}

		if(ladd[k] != 0) {
			addall(2*k+1, ladd[k]);
			addall(2*k+2, ladd[k]);
			ladd[k] = 0;
		}

		if(max_v[k] < max_v[2*k+1]) {
			update_node_max(2*k+1, max_v[k]);
		}
		if(min_v[2*k+1] < min_v[k]) {
			update_node_min(2*k+1, min_v[k]);
		}

		if(max_v[k] < max_v[2*k+2]) {
			update_node_max(2*k+2, max_v[k]);
		}
		if(min_v[2*k+2] < min_v[k]) {
			update_node_min(2*k+2, min_v[k]);
		}
	}

	void update(int k) {
		sum[k] = sum[2*k+1] + sum[2*k+2];

		if(max_v[2*k+1] < max_v[2*k+2]) {
			max_v[k] = max_v[2*k+2];
			max_c[k] = max_c[2*k+2];
			smax_v[k] = max(max_v[2*k+1], smax_v[2*k+2]);
		} else if(max_v[2*k+1] > max_v[2*k+2]) {
			max_v[k] = max_v[2*k+1];
			max_c[k] = max_c[2*k+1];
			smax_v[k] = max(smax_v[2*k+1], max_v[2*k+2]);
		} else {
			max_v[k] = max_v[2*k+1];
			max_c[k] = max_c[2*k+1] + max_c[2*k+2];
			smax_v[k] = max(smax_v[2*k+1], smax_v[2*k+2]);
		}

		if(min_v[2*k+1] < min_v[2*k+2]) {
			min_v[k] = min_v[2*k+1];
			min_c[k] = min_c[2*k+1];
			smin_v[k] = min(smin_v[2*k+1], min_v[2*k+2]);
		} else if(min_v[2*k+1] > min_v[2*k+2]) {
			min_v[k] = min_v[2*k+2];
			min_c[k] = min_c[2*k+2];
			smin_v[k] = min(min_v[2*k+1], smin_v[2*k+2]);
		} else {
			min_v[k] = min_v[2*k+1];
			min_c[k] = min_c[2*k+1] + min_c[2*k+2];
			smin_v[k] = min(smin_v[2*k+1], smin_v[2*k+2]);
		}
	}

	void _update_min(ll x, int a, int b, int k, int l, int r) {
		if(b <= l || r <= a || max_v[k] <= x) {
			return;
		}
		if(a <= l && r <= b && smax_v[k] < x) {
			update_node_max(k, x);
			return;
		}

		push(k);
		_update_min(x, a, b, 2*k+1, l, (l+r)/2);
		_update_min(x, a, b, 2*k+2, (l+r)/2, r);
		update(k);
	}

	void _update_max(ll x, int a, int b, int k, int l, int r) {
		if(b <= l || r <= a || x <= min_v[k]) {
			return;
		}
		if(a <= l && r <= b && x < smin_v[k]) {
			update_node_min(k, x);
			return;
		}

		push(k);
		_update_max(x, a, b, 2*k+1, l, (l+r)/2);
		_update_max(x, a, b, 2*k+2, (l+r)/2, r);
		update(k);
	}

	void addall(int k, ll x) {
		max_v[k] += x;
		if(smax_v[k] != -inf) smax_v[k] += x;
		min_v[k] += x;
		if(smin_v[k] != inf) smin_v[k] += x;

		sum[k] += len[k] * x;
		if(lval[k] != inf) {
			lval[k] += x;
		} else {
			ladd[k] += x;
		}
	}

	void updateall(int k, ll x) {
		max_v[k] = x; smax_v[k] = -inf;
		min_v[k] = x; smin_v[k] = inf;
		max_c[k] = min_c[k] = len[k];

		sum[k] = x * len[k];
		lval[k] = x; ladd[k] = 0;
	}

	void _add_val(ll x, int a, int b, int k, int l, int r) {
		if(b <= l || r <= a) {
			return;
		}
		if(a <= l && r <= b) {
			addall(k, x);
			return;
		}

		push(k);
		_add_val(x, a, b, 2*k+1, l, (l+r)/2);
		_add_val(x, a, b, 2*k+2, (l+r)/2, r);
		update(k);
	}

	void _update_val(ll x, int a, int b, int k, int l, int r) {
		if(b <= l || r <= a) {
			return;
		}
		if(a <= l && r <= b) {
			updateall(k, x);
			return;
		}

		push(k);
		_update_val(x, a, b, 2*k+1, l, (l+r)/2);
		_update_val(x, a, b, 2*k+2, (l+r)/2, r);
		update(k);
	}

	ll _query_max(int a, int b, int k, int l, int r) {
		if(b <= l || r <= a) {
			return -inf;
		}
		if(a <= l && r <= b) {
			return max_v[k];
		}
		push(k);
		ll lv = _query_max(a, b, 2*k+1, l, (l+r)/2);
		ll rv = _query_max(a, b, 2*k+2, (l+r)/2, r);
		return max(lv, rv);
	}

	ll _query_min(int a, int b, int k, int l, int r) {
		if(b <= l || r <= a) {
			return inf;
		}
		if(a <= l && r <= b) {
			return min_v[k];
		}
		push(k);
		ll lv = _query_min(a, b, 2*k+1, l, (l+r)/2);
		ll rv = _query_min(a, b, 2*k+2, (l+r)/2, r);
		return min(lv, rv);
	}

	ll _query_sum(int a, int b, int k, int l, int r) {
		if(b <= l || r <= a) {
			return 0;
		}
		if(a <= l && r <= b) {
			return sum[k];
		}
		push(k);
		ll lv = _query_sum(a, b, 2*k+1, l, (l+r)/2);
		ll rv = _query_sum(a, b, 2*k+2, (l+r)/2, r);
		return lv + rv;
	}

public:
	SegmentTree(int n) {
		SegmentTree(n, nullptr);
	}

	SegmentTree(int n, ll *a) : n(n) {
		n0 = 1;
		while(n0 < n) n0 <<= 1;

		for(int i=0; i<2*n0; ++i) ladd[i] = 0, lval[i] = inf;
		len[0] = n0;
		for(int i=0; i<n0-1; ++i) len[2*i+1] = len[2*i+2] = (len[i] >> 1);

		for(int i=0; i<n; ++i) {
			max_v[n0-1+i] = min_v[n0-1+i] = sum[n0-1+i] = (a != nullptr ? a[i] : 0);
			smax_v[n0-1+i] = -inf;
			smin_v[n0-1+i] = inf;
			max_c[n0-1+i] = min_c[n0-1+i] = 1;
		}

		for(int i=n; i<n0; ++i) {
			max_v[n0-1+i] = smax_v[n0-1+i] = -inf;
			min_v[n0-1+i] = smin_v[n0-1+i] = inf;
			max_c[n0-1+i] = min_c[n0-1+i] = 0;
		}
		for(int i=n0-2; i>=0; i--) {
			update(i);
		}
	}

	// range minimize query
	void update_min(int a, int b, ll x) {
		_update_min(x, a, b, 0, 0, n0);
	}

	// range maximize query
	void update_max(int a, int b, ll x) {
		_update_max(x, a, b, 0, 0, n0);
	}

	// range add query
	void add_val(int a, int b, ll x) {
		_add_val(x, a, b, 0, 0, n0);
	}

	// range update query
	void update_val(int a, int b, ll x) {
		_update_val(x, a, b, 0, 0, n0);
	}

	// range minimum query
	ll query_max(int a, int b) {
		return _query_max(a, b, 0, 0, n0);
	}

	// range maximum query
	ll query_min(int a, int b) {
		return _query_min(a, b, 0, 0, n0);
	}

	// range sum query
	ll query_sum(int a, int b) {
		return _query_sum(a, b, 0, 0, n0);
	}
};
//Segment Tree Beats end

//Prefix function start/KMP (Knuth–Morris–Pratt)
vector<int> prefix_function(string &s){
	int n=(int)s.length();
	vector<int> pi(n);
	pi[0]=0;
	for(int i=1;i<n;i++){
		int j=pi[i-1];
		while(j>0 && s[i]!=s[j]) j=pi[j-1];
		if(s[i]==s[j]) j++;
		pi[i]=j;
	}
	return pi;
}
//Prefix function end

//Z-algorithm start [Z algorithm]
vector<int> z_function(string &s){
	int n=(int)s.length();
	vector<int> z(n);
	for(int i=1,l=0,r=0; i<n; ++i){
		if(i<=r) z[i]=min(r-i+1, z[i-l]);
		while(i+z[i]<n && s[z[i]]==s[i+z[i]]) ++z[i];
		if(i+z[i]-1>r) l=i, r=i+z[i]-1;
	}
	return z;
}
//Z-algorithm end [Z algorithm]

//Trie start
struct TrieNode{
	int next[26];
	bool leaf = false;
	
    TrieNode(){fill(begin(next), end(next), -1);}
};

vector<TrieNode> Trie(1);

void addstring(const string &s){
    int v = 0;
    for(char ch : s){
		int c = ch - 'a';
		if(Trie[v].next[c] == -1){
			Trie[v].next[c] = Trie.size();
			Trie.emplace_back();
		}
		v = Trie[v].next[c];
	}
	Trie[v].leaf = true;
}
//Trie end

//DSU start
struct DSU {
	struct node{ int p; ll sz; };
	vector<node> dsu; int cc;
	node& operator[](int id){ return dsu[id]; }
	DSU(int n){ dsu.resize(n);
		forn(i,0,n){ cc=n; dsu[i].p=i; dsu[i].sz=1;}
	}
	int rt(int u){ return (dsu[u].p==u) ? u : dsu[u].p=rt(dsu[u].p); }
	bool sameset(int u, int v){ return rt(u)==rt(v); }
	void merge(int u, int v){
		u = rt(u); v = rt(v);
		if(u == v) return;
		if(dsu[u].sz < dsu[v].sz) swap(u,v);
		dsu[v].p = u;
		dsu[u].sz += dsu[v].sz;
		cc--;
	}
	ll get(int u){ return dsu[rt(u)].sz; }
	//void set(int u, ll val){ dsu[rt(u)].sz = val; }
};
//DSU end

//Kruskal start
int n,m;
vector<pair<ll,ii>> edges;
vector<pair<ii,ll>> mst;
ll sumw=0;

void kruskal()
{
	DSU dsu(n);
	sort(edges.begin(),edges.end());
	sumw=0;
	
	forn(i,0,edges.size()){
		int u=edges[i].S.F, v=edges[i].S.S; ll w=edges[i].F;
		if(dsu.sameset(u,v)) continue;
		mst.pb({{u,v},w});
		dsu.merge(u,v);
		sumw+=w;
		if(dsu.cc==1) break;
	}
}
//Kruskal end

//Dijkstra start
vii adj[MAXN];
ll dist[MAXN];

void dijkstra(int src)
{
	priority_queue<ii,vii,greater<ii>> q;
	forn(i,0,n)	dist[i]=INF;
	dist[src]=0;
	q.push({dist[src],src});
	
	while(!q.empty())
	{
		int u=q.top().S; ll curd=q.top().F; q.pop();
		if(curd>dist[u]) continue;
		for(auto tmp: adj[u])
		{
			int v=tmp.F; ll w=tmp.S;
			if(dist[v]>dist[u]+w)
			{
				dist[v]=dist[u]+w;
				q.push({dist[v],v});
			}
		}
	}
}
//Dijkstra end

//Floyd-Warshall start
ll dist[MAXN][MAXN];
void floyd(){
	forn(i,0,n) forn(j,0,n) dist[i][j] = (adj[i][j]==0 ? INF : adj[i][j]);
	forn(i,0,n) dist[i][i] = 0;
	forn(k,0,n) forn(i,0,n) forn(j,0,n)
		dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}
//Floyd-Warshall end

// SPFA/Bellman-Ford/Shortest Path Faster Algorithm start
// Returns one of the nodes in neg cycle if it exists (0-indexed), otherwise -1
int prt[MAXN];
ll dist[MAXN];

int spfa(int src)
{
	int cnt[n]{};
	bool inqueue[n]{};
	
	forn(i,0,n) dist[i]=INF, prt[i]=-1;
	dist[src]=0;
	
	queue<int> q;
	q.push(src);
	inqueue[src]=true;
	
	while(!q.empty())
	{
		int u=q.front(); q.pop();
		inqueue[u]=false;
		for(ii tmp: adj[u])
		{
			int v=tmp.F; ll w=tmp.S;
			if(dist[v]>dist[u]+w)
			{
				dist[v]=dist[u]+w;
				prt[v]=u;
				if(!inqueue[v])
				{
					q.push(v);
					inqueue[v]=true;
					cnt[v]++;
					if(cnt[v]==n)
					{
						forn(i,0,n) v=prt[v];
						return v;
					}
				}
			}
		}
	}
	
	return -1;
}
// SPFA/Bellman-Ford/Shortest Path Faster Algorithm end

//Dinic Flow start: O(V^2E)
struct DinicFlow
{
	struct Edge
	{
		int v,r; ll cap;
		Edge(int v=0, ll cap=0, int r=0): v(v), r(r), cap(cap) {}
	};
	int n_;
	vector<vector<Edge>> adj;
	vector<int> level, ptr;
	
	DinicFlow(int n)
	{
		n_ = n;
		adj.resize(n);
		level.resize(n);
		ptr.resize(n);
	}
	void addedge(int u, int v, ll c)
	{
		int u_sz = adj[u].size(), v_sz = adj[v].size();
		adj[u].emplace_back(v, c, v_sz);
		adj[v].emplace_back(u, 0, u_sz);
	}
	void bfs(int s)
	{
		level.assign(n_, -1);
		level[s] = 0;
		queue<int> q;
		q.push(s);
		while(!q.empty())
		{
			int u = q.front(); q.pop();
			for(const Edge &e: adj[u])
			{
				if(e.cap>0 && level[e.v]==-1)
				{
					level[e.v] = level[u]+1;
					q.push(e.v);
				}
			}
		}
	}
	ll dfs(int u, int t, ll f)
	{
		if(u == t) return f;
		for(int &i=ptr[u];i<adj[u].size();i++)
		{
			Edge &e = adj[u][i];
			if(e.cap>0 && level[u]+1==level[e.v])
			{
				ll newf = dfs(e.v, t, min(f,e.cap));
				if(!newf) continue;
				e.cap -= newf;
				adj[e.v][e.r].cap += newf;
				return newf;
			}
		}
		return 0;
	}
	ll flow(int s, int t)
	{
		ll sum = 0;
		while(1)
		{
			bfs(s);
			if(level[t]==-1) break;
			ptr.assign(n_, 0);
			while(1)
			{
				ll f = dfs(s, t, INF);
				if(!f) break;
				sum += f;
			}
		}
		return sum;
	}
};
//Dinic Flow end

//Min Cost Max Flow start
struct MinCostFlow
{
	struct Edge {
		int u, v; long long cap, cost;
		Edge(int u, int v, long long cap, long long cost): u(u), v(v), cap(cap), cost(cost) {}
	};
	
	int n, s, t;
	long long flow, cost;
	vector<vector<int> > graph;
	vector<Edge> e;
	vector<long long> dist, potential;
	vector<int> parent;
	bool negativeCost;
	
	MinCostFlow(int _n){
		// 0-based indexing
		n = _n;
		graph.assign(n, vector<int> ());
		negativeCost = false;
	}
	
	void addEdge(int u, int v, long long cap, long long cost, bool directed = true){
		if(cost < 0)
			negativeCost = true;
	
		graph[u].push_back(e.size());
		e.push_back(Edge(u, v, cap, cost));
	
		graph[v].push_back(e.size());
		e.push_back(Edge(v, u, 0, -cost));
	
		if(!directed)
			addEdge(v, u, cap, cost, true);
	}
	
	pair<long long, long long> getMinCostFlow(int _s, int _t){
		s = _s; t = _t;
		flow = 0, cost = 0;
	
		potential.assign(n, 0);
		if(negativeCost){
			// run Bellman-Ford to find starting potential
			dist.assign(n, 1LL<<62);
			for(int i = 0, relax = false; i < n && relax; i++, relax = false){
				for(int u = 0; u < n; u++){
					for(int k = 0; k < graph[u].size(); k++){
						int eIdx = graph[u][i];
						int v = e[eIdx].v; ll cap = e[eIdx].cap, w = e[eIdx].cost;
	
						if(dist[v] > dist[u] + w && cap > 0){
							dist[v] = dist[u] + w;
							relax = true;
			}   }   }   }
	
			for(int i = 0; i < n; i++){
				if(dist[i] < (1LL<<62)){
					potential[i] = dist[i];
		}   }   }
	
		while(dijkstra()){
			flow += sendFlow(t, 1LL<<62);
		}
	
		return make_pair(flow, cost);
	}
	
	bool dijkstra(){
		parent.assign(n, -1);
		dist.assign(n, 1LL<<62);
		priority_queue<ii, vector<ii>, greater<ii> > pq;
	
		dist[s] = 0;
		pq.push(ii(0, s));
	
	
		while(!pq.empty()){
			int u = pq.top().second;
			long long d = pq.top().first;
			pq.pop();
	
			if(d != dist[u]) continue;
	
			for(int i = 0; i < graph[u].size(); i++){
				int eIdx = graph[u][i];
				int v = e[eIdx].v; ll cap = e[eIdx].cap;
				ll w = e[eIdx].cost + potential[u] - potential[v];
	
				if(dist[u] + w < dist[v] && cap > 0){
					dist[v] = dist[u] + w;
					parent[v] = eIdx;
	
					pq.push(ii(dist[v], v));
		}   }   }
	
		// update potential
		for(int i = 0; i < n; i++){
			if(dist[i] < (1LL<<62))
				potential[i] += dist[i];
		}
	
		return dist[t] != (1LL<<62);
	}
	
	long long sendFlow(int v, long long curFlow){
		if(parent[v] == -1)
			return curFlow;
		int eIdx = parent[v];
		int u = e[eIdx].u; ll w = e[eIdx].cost;
	
		long long f = sendFlow(u, min(curFlow, e[eIdx].cap));
	
		cost += f*w;
		e[eIdx].cap -= f;
		e[eIdx^1].cap += f;
	
		return f;
	}
};
//Min Cost Max Flow end

//Hopkroft-Karp matching (MCBM, max-cardinality bipartite matching) start
//Read n1,n2 -> init() -> addEdge() -> maxMatching()
const int MAXN1 = 50000;
const int MAXN2 = 50000;
const int MAXM = 150000;

int n1, n2, edges, last[MAXN1], pre[MAXM], head[MAXM];
int matching[MAXN2], dist[MAXN1], Q[MAXN1];
bool used[MAXN1], vis[MAXN1];

void init(int _n1, int _n2) 
{
    n1 = _n1;
    n2 = _n2;
    edges = 0;
    fill(last, last + n1, -1);
}

void addEdge(int u, int v) 
{
    head[edges] = v;
    pre[edges] = last[u];
    last[u] = edges++;
}

void bfs() 
{
    fill(dist, dist + n1, -1);
    int sizeQ = 0;
    for (int u = 0; u < n1; ++u) {
        if (!used[u]) {
            Q[sizeQ++] = u;
            dist[u] = 0;
        }
    }
    for (int i = 0; i < sizeQ; i++) {
        int u1 = Q[i];
        for (int e = last[u1]; e >= 0; e = pre[e]) {
            int u2 = matching[head[e]];
            if (u2 >= 0 && dist[u2] < 0) {
                dist[u2] = dist[u1] + 1;
                Q[sizeQ++] = u2;
            }
        }
    }
}

bool dfs(int u1) 
{
    vis[u1] = true;
    for (int e = last[u1]; e >= 0; e = pre[e]) {
        int v = head[e];
        int u2 = matching[v];
        if (u2 < 0 || ((!vis[u2] && dist[u2] == dist[u1] + 1) && dfs(u2))) {
            matching[v] = u1;
            used[u1] = true;
            return true;
        }
    }
    return false;
}

int maxMatching() 
{
    fill(used, used + n1, false);
    fill(matching, matching + n2, -1);
    for (int res = 0;;) {
        bfs();
        fill(vis, vis + n1, false);
        int f = 0;
        for (int u = 0; u < n1; ++u)
            if (!used[u] && dfs(u))
                ++f;
        if (!f)
            return res;
        res += f;
    }
}
//Hopkroft-Karp matching end

//SCC (Strongly connected components) start
//init(n) -> read input -> tarjan() -> sccidx[]
struct SCC
{
	const int INF2 = int(1e9);
	vector<vector<int> > vec;
	int index;
	vector<int> idx;
	vector<int> lowlink;
	vector<bool> onstack;
	stack<int> s;
	vector<int> sccidx;
	int scccnt;
	vi topo;
	
	//lower sccidx means appear later
	void init(int n)
	{
		idx.assign(n,-1);
		index = 0;
		onstack.assign(n,0);
		lowlink.assign(n,INF2);
		while(!s.empty()) s.pop();
		sccidx.assign(n,-1);
		scccnt = 0;
		vec.clear();
		topo.clear();
		vec.resize(n);
	}
	
	void addedge(int u, int v) //u -> v
	{
		vec[u].pb(v);
	}
	
	void connect(int u)
	{
		idx[u] = index;
		lowlink[u] = index;
		index++;
		s.push(u);
		onstack[u] = true;
		for(int i = 0; i < vec[u].size(); i++)
		{
			int v = vec[u][i];
			if(idx[v] == -1)
			{
				connect(v);
				lowlink[u] = min(lowlink[u], lowlink[v]);
			}
			else if(onstack[v])
			{
				lowlink[u] = min(lowlink[u], idx[v]);
			}
		}
		if(lowlink[u] == idx[u])
		{
			while(1)
			{
				int v = s.top();
				s.pop();
				onstack[v] = false;
				sccidx[v] = scccnt;
				if(v == u) break;
			}
			scccnt++;
		}
	}
	
	void tarjan()
	{
		for(int i = 0; i < vec.size(); i++)
		{
			if(idx[i] == -1)
			{
				connect(i);
			}
		}
	}
	
	void toposort() //if graph is a DAG and i just want to toposort
	{
		tarjan();
		int n = vec.size();
		topo.resize(n);
		vector<ii> tmp;
		for(int i = 0; i < n; i++)
		{
			tmp.pb(ii(sccidx[i],i));
		}
		sort(tmp.begin(),tmp.end());
		reverse(tmp.begin(),tmp.end());
		for(int i = 0; i < n; i++)
		{
			topo[i]=tmp[i].S;
			if(i>0) assert(tmp[i].F!=tmp[i-1].F);
		}
	}
};
//SCC end

//Euler tour start
int in[MAXN],out[MAXN];
vi euler;
int tmr=-1;

void dfs_euler(int u, int p){
	in[u]=++tmr;
	euler.pb(u);
	for(int v: adj[u]){
		if(v==p) continue;
		dfs_euler(v,u);
	}
	out[u]=tmr;
}
//Euler tour end

//HLD (Heavy-light decomposition) start
#define LG 21

int in[MAXN],out[MAXN],tmr=-1;
int prt[MAXN],sz[MAXN],dep[MAXN];
int top[MAXN];

void dfs_sz(int u, int p){
	sz[u]=1;
	prt[u]=p;

	if(adj[u][0]==p && adj[u].size()>1) swap(adj[u][0],adj[u][1]);
	
	for(int &v: adj[u]){
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_sz(v,u);
		sz[u]+=sz[v];
		if(sz[v]>sz[adj[u][0]]) swap(v,adj[u][0]);
	}
}

void dfs_hld(int u, int p){
	in[u]=++tmr;
	for(int v: adj[u]){
		if(v==p) continue;
		top[v] = (v==adj[u][0]) ? top[u] : v;
		dfs_hld(v,u);
	}
	out[u]=tmr;
}

ll merge_hld(ll x, ll y){ return x+y; }
ll Query(int u, int v){
	ll ans=0;
	while(top[u]!=top[v]){
		if(dep[top[u]]<dep[top[v]]) swap(u,v);
		ans=merge_hld(ans,st.query(in[top[u]],in[u]));
		u=prt[top[u]];
	}
	
	if(dep[u]<dep[v]) swap(u,v);
	return merge_hld(ans,st.query(in[v],in[u]));
}

st.update(in[u],in[u],w);
dfs_sz(0,-1);
dfs_hld(0,-1);
//HLD (Heavy-light decomposition) end

//LCA euler O(log n) query start
const int LG = 20;

int in[MAXN],out[MAXN],tmr=-1;
int prt[LG][MAXN];
mset(prt,-1);

void dfs_lca(int u, int p)
{
	in[u]=++tmr;
	prt[0][u]=p;
	forn(i,1,LG){
		if(prt[i-1][u]!=-1) prt[i][u]=prt[i-1][prt[i-1][u]];
	}
	for(int v: adj[u]){
		if(v==p) continue;
		dfs_lca(v,u);
	}
	out[u]=tmr;
}

bool isChild(int u, int v)
{
	return in[u]<=in[v] && out[v]<=out[u];
}

int lca(int u, int v)
{
	if(isChild(u,v)) return u;
	for(int i=LG-1;i>=0;i--){
		if(prt[i][u]!=-1 && !isChild(prt[i][u],v))
			u=prt[i][u];
	}
	return prt[0][u];
}
//LCA euler O(log n) query end

//LCA depth O(log n) query start
const int LG = 20;

int dep[MAXN], prt[MAXN][LG];
mset(prt,-1); mset(dep,0);

void dfs_lca(int u, int p)
{
	prt[u][0]=p;
	forn(j,1,LG){
		if(prt[u][j-1]!=-1) prt[u][j]=prt[prt[u][j-1]][j-1];
	}
	for(int v: adj[u])
	{
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_lca(v,u);
	}
}

int lca(int u, int v)
{
	if(dep[u]>dep[v]) swap(u,v);
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[v][i]!=-1 && dep[prt[v][i]]>=dep[u])
		{
			v=prt[v][i];
		}
	}
	if(u==v) return u;
	for(int i=LG-1;i>=0;i--)
	{
		if(prt[v][i]!=-1 && prt[v][i]!=prt[u][i])
		{
			v=prt[v][i]; u=prt[u][i];
		}
	}
	return prt[u][0];
}
//LCA depth O(log n) query end

//Binary parent start
int goup(int u, int h){
	for(int i=LG-1;i>=0;i--){
		if(h&(1<<i)) u=prt[u][i];
	}
	return u;
}
//Binary parent end

//LCA O(1) query start
vi adj[MAXN];
int lg[MAXN+1];
ll spt[MAXN][LG+1];
int in[MAXN],out[MAXN],dep[MAXN];
vi euler;
int tmr=-1;

struct SparseTableLCA
{	
	ll merge(ll x, ll y){
		return (dep[x]<dep[y] ? x:y);
	}
	
	SparseTableLCA(){}	
	SparseTableLCA(vi arr){
		int N=arr.size();
		lg[1]=0;
		fore(i,2,N)	lg[i]=lg[i/2]+1;
		
		forn(i,0,N) spt[i][0] = arr[i];
		fore(j,1,LG)
			for(int i=0; i+(1<<j)<=N; i++)
				spt[i][j] = merge(spt[i][j-1], spt[i+(1<<(j-1))][j-1]);
	}
	
	ll query(int l,int r){
		int len=lg[r-l+1];		
		return merge(spt[l][len],spt[r-(1<<len)+1][len]);
	}	
}lcast;

void dfs_lca(int u, int p){
	in[u]=++tmr;
	euler.pb(u);
	for(int v: adj[u]){
		if(v==p) continue;
		dep[v]=dep[u]+1;
		dfs_lca(v,u);
		euler.pb(u);
	}
	out[u]=tmr;
}

int lca(int u, int v){
	int l=in[u],r=in[v];
	if(l>r) swap(l,r);
	return lcast.query(l,r);
}	

//in main()
dfs_lca(0,-1);
lcast=SparseTableLCA(euler);

//LCA O(1) query end

//Centroid decomposition start
int sz[MAXN];
bool vst[MAXN];

void dfs_sz(int u, int p)
{
	sz[u]=1;
	for(int v: adj[u])
	{
		if(v==p || vst[v]) continue;
		dfs_sz(v,u);
		sz[u]+=sz[v];
	}
}

void centroid(int u, int p, int r)
{
	for(int v: adj[u])
	{
		if(v==p || vst[v]) continue;
		if(sz[v]*2>sz[r]) return centroid(v,u,r);
	}
	return u;
}

void prep(int u, int p)
{
	for(int v: adj[u])
	{
		if(v==p || vst[v]) continue;
		
	}
}

void solve(int u)
{
	dfs_sz(u,-1);
	u=centroid(u,-1,u);
	
	//do stuffs
	prep(u,-1);
	for(int v: adj[u])
	{
		if(vst[v]) continue;
		
	}
	
	vst[u]=1;
	for(int v: adj[u])
	{
		if(vst[v]) continue;
		solve(v);
	}
}
//Centroid decomposition end

//Sparse Table start: O(1) Min Query
const int LG = 20;

int lg[MAXN+1];
ll spt[MAXN][LG+1];

struct SparseTable
{	
	ll merge(ll x, ll y){
		return min(x,y);
	}
	
	SparseTable(int n, ll arr[]){
		lg[1]=0;
		fore(i,2,n)	lg[i]=lg[i/2]+1;
		
		forn(i,0,n) spt[i][0] = arr[i];
		fore(j,1,LG)
			for(int i=0; i+(1<<j)<=n; i++)
				spt[i][j] = merge(spt[i][j-1], spt[i+(1<<(j-1))][j-1]);
	}
	
	ll query(int l,int r){
		int len=lg[r-l+1];		
		return merge(spt[l][len],spt[r-(1<<len)+1][len]);
	}	
};
//Sparse Table end

//Convex Hull Dynamic start (CHT)
//Source: https://github.com/kth-competitive-programming/kactl/blob/master/content/data-structures/LineContainer.h
//Finds max by default; set Max to false for min
struct Line {
	mutable ll k, m, p;
	bool operator<(const Line& o) const { return k < o.k; }
	bool operator<(ll x) const { return p < x; }
};

struct ConvexHullDynamic: multiset<Line, less<>> {
	const ll inf = LLONG_MAX; //double: inf = 1.0L
	bool Max = true;
	inline ll div(ll a, ll b){
		return a/b - ((a^b)<0 && a%b);
	}
	bool isect(iterator x, iterator y){
		if(y == end()){ x->p = inf; return false; }
		if(x->k == y->k) x->p = x->m > y->m ? inf : -inf;
		else x->p = div(y->m - x->m, x->k - y->k);
		return x->p >= y->p;
	}
	void add(ll k, ll m){ // k = slope, m = y-intercept
		if(!Max) k = -k, m = -m;
		auto z = insert({k, m, 0}), y = z++, x = y;
		while(isect(y, z)) z = erase(z);
		if(x!=begin() && isect(--x, y))
			isect(x, y = erase(y));
		while((y=x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}
	ll query(ll x){
		if(empty()) return Max ? 0 : inf;
		auto l = *lower_bound(x);
		return (l.k * x + l.m) * (Max ? 1 : -1);
	}
};
//Convex Hull Dynamic end (CHT)

//Convex Hull fast start (CHT)
struct Line {
	ll m, b;
	Line(ll _m, ll _b): m(_m), b(_b) {}
	inline ll eval(ll x){ return m*x+b; }
};

struct ConvexHull {
	deque<Line> d;
	inline void clear(){ d.clear(); }
	bool bad(const Line &Z){
		if(int(d.size())<2) return false;
		const Line &X = d[int(d.size())-2], &Y = d[int(d.size())-1];
		return (X.b-Z.b)*(Y.m-X.m) <= (X.b-Y.b)*(Z.m-X.m);
	}
	void addline(ll m, ll b){
		Line l = Line(m,b);
		while(bad(l)) d.pop_back();
		d.push_back(l);
	}
	ll query(ll x){
		if(d.empty()) return 0;
		while(int(d.size())>1 && (d[0].b-d[1].b <= x*(d[1].m-d[0].m))) d.pop_front();
		return d.front().eval(x);
	}
};
//Convex Hull fast end (CHT)

//Combi/Maths start
vector<ll> fact,ifact,inv,pow2;
ll add(ll a, ll b)
{
	a+=b; a%=MOD;
	if(a<0) a+=MOD;
	return a;
}
ll mult(ll a, ll b)
{
	if(a>MOD) a%=MOD;
	if(b>MOD) b%=MOD;
	ll ans=(a*b)%MOD;
	if(ans<0) ans+=MOD;
	return ans;
}
ll pw(ll a, ll b)
{
	ll r=1;
	while(b){
		if(b&1) r=mult(r,a);
		a=mult(a,a);
		b>>=1;
	}
	return r;
}
ll inverse(ll a)
{
	return pw(a,MOD-2);
}
ll choose(ll a, ll b)
{
	if(a<b) return 0;
	if(b==0) return 1;
	if(a==b) return 1;
	return mult(fact[a],mult(ifact[b],ifact[a-b]));
}
void init(ll _n)
{
	fact.clear(); ifact.clear(); inv.clear(); pow2.clear();
	fact.resize(_n+1); ifact.resize(_n+1); inv.resize(_n+1); pow2.resize(_n+1);
	pow2[0]=1; ifact[0]=1; fact[0]=1;
	for(int i=1;i<=_n;i++){
		pow2[i]=add(pow2[i-1],pow2[i-1]);
		fact[i]=mult(fact[i-1],i);
	}
	ifact[_n] = inverse(fact[_n]);
	for(int i=_n-1;i>=1;i--){
	    ifact[i] = mult(ifact[i+1], i+1);
	}
	for(int i=1;i<=_n;i++){
	    inv[i] = mult(fact[i-1], ifact[i]);
	}
}
void getpf(vector<ii>& pf, ll n)
{
	for(ll i=2; i*i<=n; i++)
	{
		int cnt=0;
		while(n%i==0){
			n/=i; cnt++;
		}
		if(cnt>0) pf.pb({i,cnt});
	}
	if(n>1) pf.pb({n,1});
}
//Combi/Maths end

//Matrix start
struct Matrix{
	vector<vector<ll>> a;
	vector<ll>& operator[](int x){ return a[x]; }
	inline int r(){ return a.size(); }
	inline int c(){ return (a.size() ? a[0].size() : 0); }
	
	Matrix(int r_ = 0, int c_ = 0, bool identity = 0){
		a.resize(r_, vector<ll>(c_));
		if(identity){
			assert(r_ == c_);
			for(int i = 0; i < r_; i++) a[i][i] = 1;
		}
	}
	inline Matrix(const vector<vector<ll>>& v){ a = v; }
	inline void operator=(const vector<vector<ll>>& v){ a = v; }
};
Matrix operator*(Matrix A, Matrix B){
	assert(A.c() == B.r());
	const ll MOD2 = ll(MOD) * MOD; //MOD
	Matrix C(A.r(), B.c());
	for(int i = 0; i < A.r(); i++){
		for(int j = 0; j < B.c(); j++){
			ll w = 0;
			for(int k = 0; k < A.c(); k++){
				w += ll(A[i][k]) * B[k][j];
				if(w >= MOD2) w -= MOD2; //MOD
			}
			C[i][j] = w % MOD; //MOD
		}
	}
	return C;
}
Matrix operator^(Matrix A, ll b){
	assert(A.r() == A.c());
	Matrix R = Matrix(A.r(), A.r(), 1);
	for(; b; b >>= 1){
		if(b & 1) R = R * A;
		A = A * A;
	}
	return R;
}
//Matrix end

//Number Theory NT start
vector<ll> primes, totient, sumdiv, bigdiv;
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
			for(ll j = i*i; j <= n; j += i)
			{
				prime[j] = false;
			}
		}
	}
}
ll phi(ll x)
{
	map<ll,ll> pf;
	ll num = 1; ll num2 = x;
	for(ll i = 0; primes[i]*primes[i] <= x; i++)
	{
		if(x%primes[i]==0)
		{
			num2/=primes[i];
			num*=(primes[i]-1);
		}
		while(x%primes[i]==0)
		{
			x/=primes[i];
			pf[primes[i]]++;
		}
	}
	if(x>1)
	{
		pf[x]++; num2/=x; num*=(x-1);
	}
	x = 1;
	num*=num2;
	return num;
}
bool isprime(ll x)
{
	if(x==1) return false;
	for(ll i = 0; primes[i]*primes[i] <= x; i++)
	{
		if(x%primes[i]==0) return false;
	}
	return true;
}
void SievePhi(ll n)
{
	totient.resize(n+1);
	for (int i = 1; i <= n; ++i) totient[i] = i;
	for (int i = 2; i <= n; ++i)
	{
		if (totient[i] == i)
		{
			for (int j = i; j <= n; j += i)
			{
				totient[j] -= totient[j] / i;
			}
		}
	}
}
void SieveSumDiv(ll n)
{
	sumdiv.resize(n+1);
	for(int i = 1; i <= n; ++i)
	{
		for(int j = i; j <= n; j += i)
		{
			sumdiv[j] += i;
		}
	}
}
ll getPhi(ll n)
{
	return totient[n];
}
ll getSumDiv(ll n)
{
	return sumdiv[n];
}
ll pw(ll a, ll b, ll mod)
{
	ll r = 1;
	if(b < 0) b += mod*100000LL;
	while(b)
	{
		if(b&1) r = (r*a)%mod;
		a = (a*a)%mod;
		b>>=1;
	}
	return r;
}
ll inv(ll a, ll mod)
{
	return pw(a, mod - 2, mod);
}
ll invgeneral(ll a, ll mod)
{
	ll ph = phi(mod);
	ph--;
	return pw(a, ph, mod);
}
void getpf(vector<ii>& pf, ll n)
{
	for(ll i = 0; primes[i]*primes[i] <= n; i++)
	{
		int cnt = 0;
		while(n%primes[i]==0)
		{
			n/=primes[i]; cnt++;
		}
		if(cnt>0) pf.pb(ii(primes[i], cnt));
	}
	if(n>1)
	{
		pf.pb(ii(n, 1));
	}
}
void getdiv(vector<ll>& div, vector<ii>& pf, ll n = 1, int i = 0)
{
	ll x, k;
	if(i >= pf.size()) return;
	x = n;
	for(k = 0; k <= pf[i].S; k++)
	{
		if(i==int(pf.size())-1) div.pb(x);
		getdiv(div, pf, x, i + 1);
		x *= pf[i].F;
	}
}
//End Number Theory NT

//Sqrt Decomposition/Mo's algorithm start
int BS;
struct query{
	int l,r,id;
};
bool cmp(query a, query b){
	if(a.l/BS != b.l/BS) return a.l/BS<b.l/BS;
	return a.r<b.r;
}

int n,Q;
int cur=0;
ll a[30005];
ll cnt[1000005];
query q[200005];
ll ans[200005];

void add(int p){
	if(!cnt[a[p]]) cur++;
	cnt[a[p]]++;
}

void remove(int p){
	cnt[a[p]]--;
	if(!cnt[a[p]]) cur--;
}

mset(cnt,0);
	
BS=sqrt(n);
forn(i,0,n) cin>>a[i];
cin>>Q;
forn(i,0,Q){
	int l,r; cin>>l>>r; l--; r--;
	q[i].l=l; q[i].r=r;
	q[i].id=i;
}

sort(q,q+Q,cmp);

int mo_l=0, mo_r=-1;
forn(i,0,Q){
	int L=q[i].l, R=q[i].r;
	
	while(mo_l<L) remove(mo_l++);
	while(mo_l>L) add(--mo_l);
	while(mo_r<R) add(++mo_r);
	while(mo_r>R) remove(mo_r--);
	
	ans[q[i].id]=cur;
}
//Sqrt decomposition/Mo's algorithm end

//FFT (Fast Fourier Transform) start
typedef complex<double> cd;
const double PI = acos(-1);
void fft(vector<cd> &a, bool invert)
{
	int n=a.size();
	for(int i=1,j=0;i<n;i++)
	{
		int bit=n>>1;
		for(;j&bit;bit>>=1) j^=bit;
		j^=bit;
		if(i<j) swap(a[i],a[j]);
	}
	for(int len=2;len<=n;len<<=1)
	{
		double ang=2*PI/len*(invert?-1:1);
		cd rt(cos(ang), sin(ang));
		for(int i=0;i<n;i+=len)
		{
			cd w(1);
			for(int j=0;j<len/2;j++)
			{
				cd u=a[i+j], v=w*a[i+j+len/2];
				a[i+j]=u+v;
				a[i+j+len/2]=u-v;
				w*=rt;
			}
		}
	}
	if(invert) for(cd &x: a) x/=n;
}
vi mult(vi &a, vi &b)
{
	int n=1;
	while(n<a.size()+b.size()) n<<=1;
	
	vector<cd> fa(n),fb(n);
	for(int i=0;i<a.size();i++) fa[i]=a[i];
	for(int i=0;i<b.size();i++) fb[i]=b[i];
	fft(fa,0); fft(fb,0);
	forn(i,0,n) fa[i]*=fb[i];
	fft(fa,1);
	
	vi r(n);
	for(int i=0;i<n;i++) r[i]=round(fa[i].real());
	return r;
}
//FFT (Fast Fourier Transform) end



//Randomizer start
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int>(1,6)(rng)
uniform_int_distribution<> dis(1,6)
	
	Examples:
	cout<<rng()<<'\n';
	cout<<dis(rng)<<'\n';

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
//Randomizer end

// Hash map custom hash start: unordered_map<T,T,custom_hash> mp;
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
// Hash map custom hash end

//Binary converter start
string bconv(ll x)
{
	string res;
	for(int i=8;i>=0;i--){
		if((1LL<<i)&x) res+='1';
		else res+='0';
	}
	return res;
}
//Binary converter end

//Grid movement (4-direction) start
int n,m;
const int dx[4]={-1,1,0,0};
const int dy[4]={0,0,-1,1};
const char dc[4]={'U','D','L','R'};
bool oob(int x, int y){
	return x<0 || y<0 || x>=n || y>=m;
}

bool vst[MAXN][MAXN];

forn(k,0,4){
	int x1=x+dx[k], y1=y+dy[k];
	if(oob(x1,y1)) continue;
	
	
}

void dfs(int x, int y)
{
	if(vst[x][y]) return;
	vst[x][y]=1;
	forn(k,0,4){
		int x1=x+dx[k], y1=y+dy[k];
		if(oob(x1,y1)) continue;
		if(vst[x1][y1]) continue;
		
		dfs(x1,y1);
	}
}
//Grid movement (4-direction) end

//Grid movement (8-direction) start
int dx[8]={-1,0,1,-1,1,-1,0,1};
int dy[8]={-1,-1,-1,0,0,1,1,1};
//Grid movement (8-direction) end

//Nearest/Closest pair of points start
struct Point{
	ll x,y,id;
};

ll dist(const Point &a, const Point &b){
	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}

bool cmp_x(const Point &a, const Point &b){
	return a.x<b.x || (a.x==b.x && a.y<b.y);
}
bool cmp_y(const Point &a, const Point &b){
	return a.y<b.y;
}

vector<Point> Temp;
ll mindist=INF;
ii best={-1,-1};
void rec(int l, int r, vector<Point> &a)
{
	if(r-l<=3){
		for(int i=l;i<r;i++) for(int j=i+1;j<r;j++){
			if(mindist>dist(a[i],a[j])){
				mindist=dist(a[i],a[j]);
				best={a[i].id, a[j].id};
			}
		}
		sort(a.begin()+l, a.begin()+r, cmp_y);
		return;
	}
	
	int mid=(l+r)/2;
	ll midx=a[mid].x;
	rec(l,mid,a);
	rec(mid,r,a);
	
	merge(a.begin()+l, a.begin()+mid, a.begin()+mid, a.begin()+r, Temp.begin(), cmp_y);
	copy(Temp.begin(), Temp.begin()+r-l, a.begin()+l);
	
	int sz=0;
	for(int i=l;i<r;i++){
		if(abs(a[i].x-midx)<mindist){
			for(int j=sz-1; j>=0 && a[i].y-Temp[j].y<mindist; j--){
				if(mindist>dist(a[i],Temp[j])){
					mindist=dist(a[i],Temp[j]);
					best={a[i].id, Temp[j].id};
				}
			}
			Temp[sz++]=a[i];
		}
	}
}

Temp.resize(n);
sort(a.begin(), a.end(), cmp_x);
rec(0,n,a);
//Nearest pair of points end
