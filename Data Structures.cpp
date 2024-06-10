#include <bits/stdc++.h>
//#include <ext/pb_ds/assoc_container.hpp>
//#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
//using namespace __gnu_pbds;

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2")
#define watch(x) cout<<(#x)<<"="<<(x)<<'\n'
#define mset(d,val) memset(d,val,sizeof(d))
#define cbug if(DEBUG) cout
#define setp(x) cout<<fixed<<setprecision(x)
#define sz(x) (int)(x).size()
#define all(x) begin(x), end(x)
#define forn(i,a,b) for(int i=(a);i<(b);i++)
#define fore(i,a,b) for(int i=(a);i<=(b);i++)
#define pb push_back
#define F first
#define S second
#define fbo find_by_order
#define ook order_of_key
typedef long long ll;
typedef long double ld;
typedef pair<ll,ll> ii;
typedef vector<ll> vi;
typedef vector<ii> vii;
//template<typename T>
//using pbds = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
void SD(int t=0){ cout<<"PASSED "<<t<<endl; }
ostream& operator<<(ostream &out, ii x){ out<<"("<<x.F<<","<<x.S<<")"; return out; }
const ll INF = ll(1e18);
const int MOD = 998244353;

const bool DEBUG = 0;
const int MAXN = 100005;
const int LG = 21;

// Lazy Recursive ST start
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
// Lazy recursive ST end

// Lazy recursive ST with struct start
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
// Lazy recursive ST with struct end

// Point recursive ST start
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
// Point recursive ST end

// Point recursive ST with struct start
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
// Point recursive with struct ST end

// Point iterative ST start
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
// Point iterative ST end

// 2D Segment Tree start
class SegmentTree2D {
private:
	int size_n, size_m;
	vector<vector<ll>> v;
	void build(const vector<vector<ll>> &a, int k, int l, int r)
	{
		if(r >= size_n) return;
		if(l != r){
			int mid = (l+r)>>1;
			build(a, k*2, l, mid);
			build(a, k*2+1, mid+1, r);
		}
		build2(a, k, l, r, 1, 0, size_m-1);
	}
	void build2(const vector<vector<ll>> &a, int k, int l, int r, int k2, int l2, int r2)
	{
		if(l2 == r2){
			if(l >= a.size() || l2 >= a[0].size()) return;
			if(l == r)
				v[k][k2] = a[l][l2];
			else
				v[k][k2] = merge(v[k*2][k2], v[k*2+1][k2]);
			return;
		}
		int mid2 = (l2+r2)>>1;
		build2(a, k, l, r, k2*2, l2, mid2);
		build2(a, k, l, r, k2*2+1, mid2+1, r2);
		v[k][k2] = merge(v[k][k2*2], v[k][k2*2+1]);
	}
	void update(int p1, int p2, ll val, int k, int l, int r)
	{
		if(p1 < l || r < p1) return;
		if(l != r){
			int mid = (l+r)>>1;
			update(p1, p2, val, k*2, l, mid);
			update(p1, p2, val, k*2+1, mid+1, r);
		}
		update2(p1, p2, val, k, l, r, 1, 0, size_m-1);
	}
	void update2(int p1, int p2, ll val, int k, int l, int r, int k2, int l2, int r2)
	{
		if(p2 < l2 || r2 < p2) return;
		if(l2 == r2){
			if(l == r)
				v[k][k2] ^= val;	//modification
			else
				v[k][k2] = merge(v[k*2][k2], v[k*2+1][k2]);
			return;
		}
		int mid2 = (l2+r2)>>1;
		update2(p1, p2, val, k, l, r, k2*2, l2, mid2);
		update2(p1, p2, val, k, l, r, k2*2+1, mid2+1, r2);
		v[k][k2] = merge(v[k][k2*2], v[k][k2*2+1]);
	}
	ll query(int s, int e, int s2, int e2, int k, int l, int r)
	{
		if(e < l || r < s) return 0;	//dummy value
		if(s <= l && r <= e) return query2(s2, e2, k, 1, 0, size_m-1);
		int mid = (l+r)>>1;
		ll lc = query(s, e, s2, e2, k*2, l, mid);
		ll rc = query(s, e, s2, e2, k*2+1, mid+1, r);
		return merge(lc, rc);
	}
	ll query2(int s2, int e2, int k, int k2, int l2, int r2)
	{
		if(e2 < l2 || r2 < s2) return 0;	//dummy value
		if(s2 <= l2 && r2 <= e2) return v[k][k2];
		int mid2 = (l2+r2)>>1;
		ll lc = query2(s2, e2, k, k2*2, l2, mid2);
		ll rc = query2(s2, e2, k, k2*2+1, mid2+1, r2);
		return merge(lc, rc);
	}

public:
	SegmentTree2D(): v(vector<vector<ll>>()) {}
	SegmentTree2D(int n, int m){
		for(size_n=1;size_n<n;) size_n<<=1;
		for(size_m=1;size_m<m;) size_m<<=1;
		v.resize(4*size_n, vector<ll>(4*size_m));
	}
	inline ll merge(ll x, ll y){
		return x+y;
	}
	inline void build(const vector<vector<ll>> &a){
		build(a, 1, 0, size_n-1);
	}
	inline void update(int p1, int p2, ll val){
		update(p1, p2, val, 1, 0, size_n-1);
	}
	inline ll query(int l, int r, int l2, int r2){
		return query(l, r, l2, r2, 1, 0, size_n-1);
	}
};
// 2D Segment Tree end

// Persistent Segment Tree start
template<class T>
struct Node
{
    T val;
    int l = 0, r = 0;
};

template<class T, T invVal>
class PersistSegmentTree
{
private:
    int update(int pos, T val, int k, int l, int r) {
        int k1 = createNode();
        v[k1] = v[k];
        if (l == r) {
            v[k1].val += val;   // change this update
            return k1;
        }
        int mid = (l + r) >> 1;
        int cl = v[k1].l;
        int cr = v[k1].r;
        if (pos <= mid) {
            v[k1].l = update(pos, val, cl, l, mid);
            cl = v[k1].l;
        } else {
            v[k1].r = update(pos, val, cr, mid + 1, r);
            cr = v[k1].r;
        }
        v[k1].val = merge(v[cl].val, v[cr].val);
        return k1;
    }
    T query(int s, int e, int k, int l, int r) {
    	if (r < s || e < l) return invVal;
    	if (s <= l && r <= e) return v[k].val;
    	int mid = (l + r) >> 1;
    	T cl = query(s, e, v[k].l, l, mid);
    	T cr = query(s, e, v[k].r, mid + 1, r);
    	return merge(cl, cr);
    }

public:
    int siz;
    vector<Node<T>> v;
    PersistSegmentTree(int n) : siz(1), v(1, {invVal}) {
        while (siz < n) siz <<= 1;
    }
    int createNode() {
        v.push_back({T{}, 0, 0});
        return sz(v) - 1;
    }
    T merge(T x, T y) {
        return x + y;
    }
    int update(int pos, T val, int node) {
        return update(pos, val, node, 0, siz - 1);
    }
    T query(int l, int r, int node) {
    	return query(l, r, node, 0, siz - 1);
    }
};
// Persistent Segment Tree end

// Segment Tree Beats start (by yaketake08/tjake)
// https://tjkendev.github.io/procon-library/cpp/range_query/segment_tree_beats_2.html
// All intervals are [L,R)

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
// Segment Tree Beats end

// Fenwick Tree (FenwickPoint) start
struct FenwickPoint
{
	vector<ll> fw;
	int siz;
	FenwickPoint(): fw(vector<ll>()), siz(0) {}
	FenwickPoint(int N){ fw.assign(N+1,0); siz = N+1; }
	void reset(int N){ fw.assign(N+1,0); siz = N+1; }
	void add(int p, ll val)
	{
		for(p++; p<siz; p+=(p&(-p))) fw[p]+=val;
	}
	ll sum(int p)
	{
		ll res=0;
		for(; p; p-=(p&(-p))) res+=fw[p];
		return res;
	}
	ll query(int l, int r)
	{
		l++; r++;
		if(r<l) return 0;
		if(l==0) return sum(r);
		return sum(r)-sum(l-1);
	}
	inline void modify(int p, ll val){ add(p, val-query(p,p)); }
};
// Fenwick Tree (FenwickPoint) end

// FenwickRange start
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
		for(int tl=l; tl<siz; tl+=(tl&(-tl))) fw[tl]+=val, fw2[tl]-=val*ll(l-1);
		for(int tr=r+1; tr<siz; tr+=(tr&(-tr))) fw[tr]-=val, fw2[tr]+=val*ll(r);
	}
	ll sum(int r) //[1,r]
	{
		ll res=0;
		for(int tr=r; tr; tr-=(tr&(-tr))) res+=fw[tr]*ll(r)+fw2[tr];
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
// FenwickRange end

// Prefix function start/KMP (Knuth–Morris–Pratt)
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
// Prefix function end

// Z-algorithm/Z-function start [Z algorithm/Z function]
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
// Z-algorithm end

// Trie start
struct TrieNode {
	int next[26];
	bool leaf = false;
	TrieNode(){fill(begin(next), end(next), -1);}
};
struct Trie {
	int siz;
	vector<TrieNode> tr;
	Trie(): siz(0), tr(vector<TrieNode>(1)) {}
	TrieNode& operator[](int u){ return tr[u]; }
	int size(){ return siz; }
	void addstring(const string &s) {
		int v = 0;
		for (char ch : s) {
			int c = ch - 'a';
			if (tr[v].next[c] == -1) {
				tr[v].next[c] = tr.size();
				tr.emplace_back();
			}
			v = tr[v].next[c];
		}
		if(!tr[v].leaf) siz++;
		tr[v].leaf = true;
	}
	template<class F>
	void dfs(int u, F f) {
		forn(i,0,26) {
			if(tr[u].next[i] != -1) {
				dfs(tr[u].next[i]);
			}
		}
	}
};
// Trie end

// Aho-Corasick start [Aho Corasick]
// Reference: https://codeforces.com/blog/entry/14854
struct AhoCorasick {
	enum {alpha = 26, first = 'A'}; // change this!
	struct Node {
		int fail = 0;
        int next[alpha];
        bool leaf = 0;
		Node() { memset(next, 0, sizeof(next)); }
	};
	vector<Node> N;
	void insert(const string& s) {
		assert(!s.empty());
		int u = 0;
		for (char c : s) {
			int nxt = N[u].next[c - first];
			if (!nxt) {
                nxt = N[u].next[c - first] = sz(N);
                N.emplace_back();
            }
			u = nxt;
		}
        N[u].leaf = 1;
	}
	AhoCorasick(const vector<string>& v) : N(1) {
		forn(i,0,sz(v)) insert(v[i]);

		queue<int> q;
        forn(i,0,alpha) if (N[0].next[i]) q.push(N[0].next[i]);
		for (; !q.empty(); q.pop()) {
			int u = q.front(), prev = N[u].fail;
			forn(i,0,alpha) {
				int& nxt = N[u].next[i];
                int y = N[prev].next[i];
				if (!nxt) nxt = y;
				else {
					N[nxt].fail = y;
					q.push(nxt);
				}
			}
		}
	}
};
// Aho-Corasick end

// Suffix array start
const int MAX_N = 500005;
const int MAX_C = 305; // Maximum initial rank + 1

int RA[MAX_N], tempRA[MAX_N]; // rank array and temporary rank array
int SA[MAX_N], tempSA[MAX_N]; // suffix array and temporary suffix array. SA[i] = starting position of i-th suffix in lexicographical order
int RSA[MAX_N]; // RSA[i]: the i-th suffix is in the RSA[i]-ith lexicographically smallest suffix
int SRT[MAX_N]; // for counting/radix sort
int Phi[MAX_N], LCP[MAX_N], PLCP[MAX_N]; // for LCP counting

class SuffixArray {
public:
	vector<int> T; // the input string, up to MAX_N characters
	int n; // the length of input string
	void countingSort(int k) { // O(n)
		int i, sum, maxi = max(MAX_C, n); // up to 255 ASCII chars or length of n
		memset(SRT, 0, sizeof SRT); // clear frequency table
		for (i = 0; i < n; i++) // count the frequency of each integer rank
			SRT[i + k < n ? RA[i + k] : 0]++;
		for (i = sum = 0; i < maxi; i++) {
			int t = SRT[i]; SRT[i] = sum; sum += t;
		}
		for (i = 0; i < n; i++) // shuffle the suffix array if necessary
			tempSA[SRT[SA[i] + k < n ? RA[SA[i] + k] : 0]++] = SA[i];
		for (i = 0; i < n; i++) // update the suffix array SA
			SA[i] = tempSA[i];
	}
	void constructSA(const vector<int> &s) { // this version can go up to 100000 characters
		int i, k, r;
		T = s;
		n = s.size();
		for (i = 0; i < n; i++) RA[i] = T[i]; // initial rankings
		for (i = 0; i < n; i++) SA[i] = i; // initial SA: {0, 1, 2, ..., n-1}
		for (k = 1; k < n; k <<= 1) { // repeat sorting process log n times
			countingSort(k); // actually radix sort: sort based on the second item
			countingSort(0); // then (stable) sort based on the first item
			tempRA[SA[0]] = r = 0; // re-ranking; start from rank r = 0
			for (i = 1; i < n; i++) // compare adjacent suffixes
				tempRA[SA[i]] = // if same pair => same rank r; otherwise, increase r
				(RA[SA[i]] == RA[SA[i - 1]] && RA[SA[i] + k] == RA[SA[i - 1] + k]) ? r : ++r;
			for (i = 0; i < n; i++) // update the rank array RA
				RA[i] = tempRA[i];
			if (RA[SA[n - 1]] == n - 1) break; // nice optimization trick
		}
	}
	void computeLCP() {
		int i, L;
		Phi[SA[0]] = -1; // default value
		for (i = 1; i < n; i++) // compute Phi in O(n)
			Phi[SA[i]] = SA[i - 1]; // remember which suffix is behind this suffix
		for (i = L = 0; i < n; i++) { // compute Permuted LCP in O(n)
			if (Phi[i] == -1) { PLCP[i] = 0; continue; } // special case
			while (T[i + L] == T[Phi[i] + L]) L++; // L increased max n times
			PLCP[i] = L;
			L = max(L - 1, 0); // L decreased max n times
		}
		for (i = 0; i < n; i++) // compute LCP in O(n)
			LCP[i] = PLCP[SA[i]]; // put the permuted LCP to the correct position
	}
	void constructRSA() {
		int i;
		for (i = 0; i < n; i++) RSA[SA[i]] = i;
	}
};
// Suffix array end

// DSU start
struct DSU {
	struct Node{ int p, sz; };
	vector<Node> dsu; int cc;
	Node& operator[](int id){ return dsu[rt(id)]; }
	DSU(int n){ dsu.resize(n);
		forn(i,0,n){ cc=n; dsu[i]={i,1}; }
	}
	inline int rt(int u){ return (dsu[u].p==u) ? u : dsu[u].p=rt(dsu[u].p); }
	inline bool sameset(int u, int v){ return rt(u)==rt(v); }
	void merge(int u, int v){
		u = rt(u); v = rt(v);
		if(u == v) return;
		if(dsu[u].sz < dsu[v].sz) swap(u,v);
		dsu[v].p = u;
		dsu[u].sz += dsu[v].sz;
		cc--;
	}
};
// DSU end

// Kruskal start
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
// Kruskal end

// Dijkstra start
vector<ii> adj[MAXN];  // (node, distance)
ll dist[MAXN];
// int parents[MAXN];

void dijkstra(int src)
{
	priority_queue<ii, vector<ii>, greater<ii>> q; // (distance, node)
	fill(dist, dist + n, INF);
	// fill(parents, parents + n, -1);
	dist[src] = 0;
	q.push({dist[src], src});
	while (!q.empty())
	{
		auto [cur_dist, u] = q.top();
		q.pop();
		if (cur_dist > dist[u]) continue;
		for (auto [v, w] : adj[u])
		{
			if (dist[v] <= cur_dist + w) continue;
			dist[v] = cur_dist + w;
			// parents[v] = u;
			q.push({dist[v], v});
		}
	}
}
// Dijkstra end

// Floyd-Warshall start
ll dist[MAXN][MAXN];
void floyd(){
	forn(i,0,n) forn(j,0,n) dist[i][j] = (adj[i][j]==0 ? INF : adj[i][j]);
	forn(i,0,n) dist[i][i] = 0;
	forn(k,0,n) forn(i,0,n) forn(j,0,n)
		dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}
// Floyd-Warshall end

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

// Dinic Flow start: O(V^2E)
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
// Dinic Flow end

// Min Cost Max Flow start
struct Edge {
	int u, v; long long cap, cost;
	Edge(int u, int v, long long cap, long long cost): u(u), v(v), cap(cap), cost(cost) {}
};
struct MinCostFlow
{
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
// Min Cost Max Flow end

// Min Cost Max Flow (long double) start
typedef pair<ld,int> dii;
struct Edge {
	int u, v; ld cap, cost;
	Edge(int u, int v, ld cap, ld cost): u(u), v(v), cap(cap), cost(cost) {}
};
struct MinCostFlow
{
	int n, s, t;
	ld flow, cost;
	vector<vector<int> > graph;
	vector<Edge> e;
	vector<ld> dist, potential;
	vector<int> parent;
	bool negativeCost;

	MinCostFlow(int _n){
		// 0-based indexing
		n = _n;
		graph.assign(n, vector<int> ());
		negativeCost = false;
	}

	void addEdge(int u, int v, ld cap, ld cost, bool directed = true){
		if(cost < 0)
			negativeCost = true;

		graph[u].push_back(e.size());
		conv[{u,v}]=e.size();
		e.push_back(Edge(u, v, cap, cost));

		graph[v].push_back(e.size());
		e.push_back(Edge(v, u, 0, -cost));

		if(!directed)
			addEdge(v, u, cap, cost, true);
	}

	pair<ld, ld> getMinCostFlow(int _s, int _t){
		s = _s; t = _t;
		flow = 0, cost = 0;

		potential.assign(n, 0);
		if(negativeCost){
			// run Bellman-Ford to find starting potential
			dist.assign(n, 1e10);
			for(int i = 0, relax = false; i < n && relax; i++, relax = false){
				for(int u = 0; u < n; u++){
					for(int k = 0; k < sz(graph[u]); k++){
						int eIdx = graph[u][i];
						int v = e[eIdx].v; ld cap = e[eIdx].cap, w = e[eIdx].cost;

						if(dist[v] > dist[u] + w && cap > 1e-9){
							dist[v] = dist[u] + w;
							relax = true;
			}   }   }   }

			for(int i = 0; i < n; i++){
				if(dist[i] < (1e10)){
					potential[i] = dist[i];
		}   }   }

		while(dijkstra()){
			flow += sendFlow(t, 1e10);
		}

		return make_pair(flow, cost);
	}

	bool dijkstra(){
		parent.assign(n, -1);
		dist.assign(n, 1e10);
		priority_queue<dii, vector<dii>, greater<dii> > pq;

		dist[s] = 0;
		pq.push(dii(0, s));


		while(!pq.empty()){
			int u = pq.top().second;
			ld d = pq.top().first;
			pq.pop();

			if(d != dist[u]) continue;

			for(int i = 0; i < sz(graph[u]); i++){
				int eIdx = graph[u][i];
				int v = e[eIdx].v; ld cap = e[eIdx].cap;
				ld w = e[eIdx].cost + potential[u] - potential[v];

				if(dist[u] + w < dist[v] && cap > 1e-9){
					dist[v] = dist[u] + w;
					parent[v] = eIdx;

					pq.push(dii(dist[v], v));
		}   }   }

		// update potential
		for(int i = 0; i < n; i++){
			if(dist[i] < (1e10))
				potential[i] += dist[i];
		}

		return dist[t] != (1e10);
	}

	ld sendFlow(int v, ld curFlow){
		if(parent[v] == -1)
			return curFlow;
		int eIdx = parent[v];
		int u = e[eIdx].u; ld w = e[eIdx].cost;

		ld f = sendFlow(u, min(curFlow, e[eIdx].cap));

		cost += f*w;
		e[eIdx].cap -= f;
		e[eIdx^1].cap += f;

		return f;
	}
};
// Min Cost Max Flow (long double) end

// Hopcroft-Karp matching (MCBM, max-cardinality bipartite matching) start
// Read n1,n2 -> init() -> addEdge() -> maxMatching()
const int MAXN1 = 50000;
const int MAXN2 = 50000;
const int MAXM = 150000;

int n1, n2, edges, last[MAXN1], pre[MAXM], head[MAXM];
int matching[MAXN2], dist[MAXN1], Q[MAXN1];
bool used[MAXN1], vis[MAXN1];

class HopcroftKarp {
public:
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
};
// Hopcroft-Karp matching end

// SCC (Strongly connected components) start
// init(n) -> read input -> tarjan() -> sccidx[]
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
	vector<vector<int>> adj; // condensation graph
	int scccnt;
	vi topo;

	// lower sccidx means appear later
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
		for(int i = 0; i < sz(vec[u]); i++)
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
		for(int i = 0; i < sz(vec); i++)
		{
			if(idx[i] == -1)
			{
				connect(i);
			}
		}
	}
	void condense() // run after tarjan
	{
		adj.resize(scccnt);
		for(int u = 0; u < sz(vec); u++)
		{
			for(int v: vec[u])
			{
				if(sccidx[u] != sccidx[v])
				{
					adj[sccidx[u]].push_back(sccidx[v]);
				}
			}
		}
		for(int u = 0; u < scccnt; u++)
		{
			sort(adj[u].begin(), adj[u].end());
			adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());
		}
	}
	void toposort() // if graph is a DAG and i just want to toposort
	{
		tarjan();
		int n = sz(vec);
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
// SCC end

// HLD (Heavy-light decomposition) start
int in[MAXN],out[MAXN],tmr=-1;
int prt[MAXN],sz[MAXN],dep[MAXN];
int top[MAXN];

void dfs_sz(int u, int p)
{
	sz[u] = 1;
	prt[u] = p;
	if(sz(adj[u])>1 && adj[u][0]==p) swap(adj[u][0], adj[u][1]);

	for(auto &v: adj[u])
	{
		if(v == p) continue;
		dep[v] = dep[u] + 1;
		dfs_sz(v, u);
		sz[u] += sz[v];
		if(sz[v] > sz[adj[u][0]]) swap(v, adj[u][0]);
	}
}
void dfs_hld(int u, int p)
{
	if(p == -1) top[u] = u;
	in[u] = ++tmr;
	for(int v: adj[u])
	{
		if(v == p) continue;
		top[v] = (v == adj[u][0]) ? top[u] : v;
		dfs_hld(v, u);
	}
	out[u] = tmr;
}
inline void init_hld(int rt){ dfs_sz(rt, -1); dfs_hld(rt, -1); }
inline ll merge_hld(ll x, ll y){ return x + y; }
ll Query(int u, int v)
{
	ll ans = 0; // dummy value
	while(top[u] != top[v])
	{
		if(dep[top[u]] < dep[top[v]]) swap(u, v);
		ans = merge_hld(ans, st.query(in[top[u]], in[u]));
		u = prt[top[u]];
	}
	if(dep[u] < dep[v]) swap(u, v);
	return merge_hld(ans, st.query(in[v], in[u]));
}
// For lazy segtree, untested (I can't find problems to test)
void Update(int u, int v, ll val)
{
	while(top[u] != top[v])
	{
		if(dep[top[u]] < dep[top[v]]) swap(u, v);
		st.update(in[top[u]], in[u], val);
		u = prt[top[u]];
	}
	if(dep[u] < dep[v]) swap(u, v);
	st.update(in[v], in[u], val);
}

// init_hld(root) -> update/queries
// Point update: st.update(in[u],w);
// Update, Query: Range update/query on the path between u, v
// HLD (Heavy-light decomposition) end

// LCA euler O(log n) query start
const int LG = 21;

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
int getLca(int u, int v)
{
	if(isChild(u,v)) return u;
	for(int i=LG-1;i>=0;i--){
		if(prt[i][u]!=-1 && !isChild(prt[i][u],v))
			u=prt[i][u];
	}
	return prt[0][u];
}
// LCA euler O(log n) query end

// LCA depth O(log n) query start
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
// LCA depth O(log n) query end

// Binary parent start
int goup(int u, int h){
	for(int i=LG-1;i>=0;i--){
		if(h&(1<<i)) u=prt[u][i];
	}
	return u;
}
// Binary parent end

// LCA O(1) query start
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

// in main()
dfs_lca(0,-1);
lcast=SparseTableLCA(euler);

// LCA O(1) query end

// Centroid decomposition start
int sz[MAXN];
bool vst[MAXN];
int cprt[MAXN]; // centroid tree parent
vector<int> child[MAXN]; // subtree of centroid tree
mset(cprt,-1);

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
int centroid(int u, int p, int r)
{
	for(int v: adj[u])
	{
		if(v==p || vst[v]) continue;
		if(sz[v]*2>sz[r]) return centroid(v,u,r);
	}
	return u;
}
int build_tree(int u)
{
	dfs_sz(u,-1);
	u=centroid(u,-1,u);
	vst[u]=1;
	for(int v: adj[u])
	{
		if(vst[v]) continue;
		cprt[build_tree(v)]=u;
	}
	return u;
}
void prep(int u, int p)
{
	for(int v: adj[u])
	{
		if(v==p || vst[v]) continue;

		prep(v, u);
	}
}
void solve(int u)
{
	dfs_sz(u,-1);
	u=centroid(u,-1,u);

	prep(u,-1);
	for(int v: adj[u])
	{
		if(vst[v]) continue;

	}

	// do stuffs

	vst[u]=1;
	for(int v: adj[u])
	{
		if(vst[v]) continue;
		solve(v);
	}
}
// Centroid decomposition end

// Virtual tree start
// Modifies vadj
int buildVirtualTree(vector<int> nodes, vi vadj[])
{
	// Change these as needed
	auto reset = [&](int u) {
		vadj[u].clear();
	};
	auto connect = [&](int u, int v) {  // u is parent of v
		vadj[u].push_back(v);
	};

	auto cmpDfs = [&](int u, int v) {
		return in[u] < in[v];
	};
	sort(nodes.begin(), nodes.end(), cmpDfs);
	unordered_set<int> uniqueNodes(nodes.begin(), nodes.end());
	for (int i{1}; i < sz(nodes); i++)
		uniqueNodes.insert(getLca(nodes[i - 1], nodes[i]));
	nodes = vector<int>(uniqueNodes.begin(), uniqueNodes.end());
	sort(nodes.begin(), nodes.end(), cmpDfs);
	for_each(nodes.begin(), nodes.end(), reset);

	stack<int> stk;
    for (int u : nodes)
	{
		if (stk.empty()) { stk.push(u); continue; }
		while (!isChild(stk.top(), u)) stk.pop();
		connect(stk.top(), u);
		stk.push(u);
	}
	return nodes[0];
}
// Virtual tree end

// Sparse Table start: O(1) Max Query
struct SparseTable
{
	vector<int> lg;
	vector<vector<ll>> spt;

	SparseTable(){}
	SparseTable(int n, ll arr[]){
		lg.resize(n + 1);
		spt.resize(n, vector<ll>(LG + 1));
		lg[1]=0;
		fore(i,2,n)	lg[i] = lg[i/2] + 1;
		forn(i,0,n) spt[i][0] = arr[i];
		fore(j,1,LG)
			for(int i=0; i+(1<<j)<=n; i++)
				spt[i][j] = merge(spt[i][j-1], spt[i+(1<<(j-1))][j-1]);
	}
	ll query(int l, int r){
		int len = lg[r-l+1];
		return merge(spt[l][len], spt[r-(1<<len)+1][len]);
	}
	inline ll merge(ll x, ll y){
		return max(x,y);
	}
};
// Sparse Table end

// Convex Hull Dynamic start (CHT)
// Source: https://github.com/kth-competitive-programming/kactl/blob/master/content/data-structures/LineContainer.h
// Finds max by default; set Max to false for min
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
// Convex Hull Dynamic end (CHT)

// Li Chao Tree start
struct Line {
	ll m,c;
	Line(): m(0), c(INF) {}
	Line(ll m, ll c): m(m), c(c) {}
	ll eval(ll x){ return m*x+c; }
};

struct LiChaoTree {
	int sz;
	bool isMax; // whether this maintains max
	vector<Line> v;
	LiChaoTree(): sz(0), isMax(false), v(vector<Line>()) {}
	LiChaoTree(int sz, bool isMax): sz(sz), isMax(isMax) {
		v.resize(sz*4, {0,INF});
	}
	void addline(Line& val) {
		if(isMax) {
			val.m = -val.m;
			val.c = -val.c;
		}
		addline(val, 1, 0, sz-1);
	}
	ll query(int x) {
		return (isMax ? -1 : 1) * query(x, 1, 0, sz-1);
	}
	void addline(Line& val, int k, int l, int r) {
		int mid = (l+r)>>1;
		bool lc = val.eval(l) <= v[k].eval(l);
		bool mc = val.eval(mid) <= v[k].eval(mid);
		if(mc) swap(val, v[k]);
		if(l==r) return;
		if(lc==mc) addline(val, k*2, l, mid);
		else addline(val, k*2+1, mid+1, r);
	}
	ll query(int x, int k, int l, int r) {
		ll cur = v[k].eval(x);
		if(l==r) return cur;
		int mid=(l+r)>>1;
		if(x<=mid) return min(cur, query(x, k*2, l, mid));
		return min(cur, query(x, k*2+1, mid+1, r));
	}
};
// Li Chao Tree end

// Convex Hull fast start (CHT)
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
// Convex Hull fast end (CHT)

// Combi/Maths start
vector<ll> fact,ifact,inv,pow2;
ll add(ll a, ll b, ll m = MOD)
{
	a+=b;
	if(abs(a)>=m) a%=m;
	if(a<0) a+=m;
	return a;
}
ll mult(ll a, ll b, ll m = MOD)
{
	if(abs(a)>=m) a%=m;
	if(abs(b)>=m) b%=m;
	a*=b;
	if(abs(a)>=m) a%=m;
	if(a<0) a+=m;
	return a;
}
void radd(ll &a, ll b, ll m = MOD){ a=add(a,b,m); }
void rmult(ll &a, ll b, ll m = MOD){ a=mult(a,b,m); }
ll pw(ll a, ll b, ll m = MOD)
{
	assert(b >= 0);  // can return 0 if desired
	if(abs(a)>=m) a%=m;
	if(a==0 && b==0) return 0; // value of 0^0
	ll r=1;
	while(b){
		if(b&1) r=mult(r,a,m);
		a=mult(a,a,m);
		b>>=1;
	}
	return r;
}
ll inverse(ll a, ll m = MOD)
{
	return pw(a,m-2);
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
// partition n into k blocks of size >= 0
ll nonneg_partition(ll n, ll k)
{
	assert(k >= 1);  // can return 0 if desired
	return choose(n + k - 1, k - 1);
}
// partition n into k blocks of size >= minVal
ll partition(ll n, ll k, ll minVal = 1)
{
	assert(k >= 1);  // can return 0 if desired
	return nonneg_partition(n - k * minVal, k);
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
// Combi/Maths end

// Matrix start
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
// Matrix end

// Number Theory NT start
vector<ll> primes, totient, sumdiv, bigdiv, lowprime, mobius;
vector<bool> isprime;
void Sieve(ll n) // linear Sieve
{
	isprime.assign(n+1, 1);
	lowprime.assign(n+1, 0);
	isprime[1] = false;
	for(ll i = 2; i <= n; i++)
	{
		if(lowprime[i] == 0)
		{
			primes.pb(i);
			lowprime[i] = i;
		}
		for(int j=0; j<sz(primes) && primes[j]<=lowprime[i] && i*primes[j]<=n; j++)
		{
			isprime[i*primes[j]] = false;
			lowprime[i*primes[j]] = primes[j];
		}
	}
}
void SieveMobius(ll n)
{
	mobius.resize(n + 1);
	mobius[1] = 1;
	for (ll i = 2; i <= n; i++)
    {
		if (lowprime[i] == i) mobius[i] = -1;
		for (int j = 0; j < sz(primes) && primes[j] <= lowprime[i] && i * primes[j] <= n; j++)
        {
            ll cur = i * primes[j];
            if (primes[j] == lowprime[i]) mobius[cur] = 0;
            else mobius[cur] = -mobius[i];
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
	if (pf.empty())  // divisors of 1
	{
		div = {1};
		return;
	}
	ll x, k;
	if(i >= sz(pf)) return;
	x = n;
	for(k = 0; k <= pf[i].S; k++)
	{
		if(i == sz(pf) - 1) div.pb(x);
		getdiv(div, pf, x, i + 1);
		x *= pf[i].F;
	}
}
// End Number Theory NT

// Longest increasing subsequence (lis) start
ll lisend[MAXN];
int lislen[MAXN], idx[MAXN], prt[MAXN];
int lis(int n, ll a[])
{
	const ll inf = 2e9;

	lisend[0]=-inf;
	for(int i=1;i<n;i++) lisend[i]=inf;
	idx[0]=prt[0]=-1;

	int maxlen=0;
	for(int i=0;i<n;i++)
	{
		int p=upper_bound(lisend, lisend+n, a[i])-lisend;
		lisend[p]=a[i];
		idx[p]=i;
		prt[p]=idx[p-1];
		maxlen=max(maxlen, p);
		lislen[i]=maxlen;
	}
	return maxlen;
}
// Longest increasing subsequence (lis) end

// Sqrt decomposition/Mo's algorithm start
const int BS;
struct Query {
	int l,r,id;
	inline ii toPair() const {
		return {l/BS, ((l/BS)&1)?-r:r};
	}
};
inline bool operator<(const Query &a, const Query &b) {
	return a.toPair() < b.toPair();
}

void add(int p)
{

}

void remove(int p)
{

}

int L=0,R=-1;
for(int i=0;i<Q;i++)
{
	int ql=q[i].l, qr=q[i].r, id=q[i].id;
	while(L>ql)
	{
		add(--L);
	}
	while(R<qr)
	{
		add(++R);
	}
	while(L<ql)
	{
		remove(L++);
	}
	while(R>qr)
	{
		remove(R--);
	}
}
// Sqrt decomposition/Mo's algorithm end

// FFT (Fast Fourier Transform) start
typedef complex<ld> cd;
const ld PI = acos(-1);
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
		ld ang=2*PI/len*(invert?-1:1);
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
// FFT (Fast Fourier Transform) end

// NTT (Number Theoretic Transform) start

// Source: https://cp-algorithms.com/algebra/fft.html
//const int MOD = 998244353;
const int ROOT = 3; // primitive root
const int ROOT_1 = 2446678; // ROOT's inverse
const int ROOT_PW = 1 << 23;
// copy add/mult/inverse from math/combi
void ntt(vi &a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = invert ? ROOT_1 : ROOT;
        for (int i = len; i < ROOT_PW; i <<= 1)
            wlen = (int)(1LL * wlen * wlen % MOD);

        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; j++) {
                int u = a[i+j], v = (int)(1LL * a[i+j+len/2] * w % MOD);
                a[i+j] = u + v < MOD ? u + v : u + v - MOD;
                a[i+j+len/2] = u - v >= 0 ? u - v : u - v + MOD;
                w = (int)(1LL * w * wlen % MOD);
            }
        }
    }

    if (invert) {
        int n_1 = inverse(n);
        for (int & x : a)
            x = (int)(1LL * x * n_1 % MOD);
    }
}
vi mult(vi &a, vi &b)
{
	int n=1;
	while(n<sz(a)+sz(b)) n<<=1;

	vi fa(n),fb(n);
	for(int i=0;i<sz(a);i++) fa[i]=a[i];
	for(int i=0;i<sz(b);i++) fb[i]=b[i];
	ntt(fa, 0); ntt(fb, 0);
	forn(i,0,n) fa[i]*=fb[i];
	ntt(fa, 1);

	vi r(n);
	for(int i=0;i<n;i++) r[i]=round(fa[i].real());
	return r;
}
// NTT (Number Theoretic Transform) end

// FFT mod start
// Usage: res = convMod<MOD>(A, B);

// Source: http://neerc.ifmo.ru/trains/toulouse/2017/fft2.pdf
typedef complex<double> CD;
void fft(vector<CD>& a) {
	int n = a.size(), L = 31 - __builtin_clz(n);
	static vector<complex<long double>> R(2, 1);
	static vector<CD> rt(2, 1);  // (^ 10% faster if double)
	for (static int k = 2; k < n; k *= 2) {
		R.resize(n); rt.resize(n);
		auto x = polar(1.0L, acos(-1.0L) / k);
		forn(i,k,2*k) rt[i] = R[i] = i&1 ? R[i/2] * x : R[i/2];
	}
	vector<int> rev(n);
	forn(i,0,n) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
	forn(i,0,n) if (i < rev[i]) swap(a[i], a[rev[i]]);
	for (int k = 1; k < n; k *= 2)
		for (int i = 0; i < n; i += 2*k) forn(j,0,k) {
			// CD z = rt[j+k] * a[i+j+k]; // (25% faster if hand-rolled)  /// include-line
			auto x = (double *)&rt[j+k], y = (double *)&a[i+j+k];		/// exclude-line
			CD z(x[0]*y[0] - x[1]*y[1], x[0]*y[1] + x[1]*y[0]);		   /// exclude-line
			a[i + j + k] = a[i + j] - z;
			a[i + j] += z;
		}
}

typedef vector<ll> vl;
template<int M> vl convMod(const vl &a, const vl &b) {
	if (a.empty() || b.empty()) return {};
	vl res(a.size() + b.size() - 1);
	int B=32-__builtin_clz((int)res.size()), n=1<<B, cut=int(sqrt(M));
	vector<CD> L(n), R(n), outs(n), outl(n);
	forn(i,0,a.size()) L[i] = CD((int)a[i] / cut, (int)a[i] % cut);
	forn(i,0,b.size()) R[i] = CD((int)b[i] / cut, (int)b[i] % cut);
	fft(L), fft(R);
	forn(i,0,n) {
		int j = -i & (n - 1);
		outl[j] = (L[i] + conj(L[j])) * R[i] / (2.0 * n);
		outs[j] = (L[i] - conj(L[j])) * R[i] / (2.0 * n) / 1i;
	}
	fft(outl), fft(outs);
	forn(i,0,res.size()) {
		ll av = ll(real(outl[i])+.5), cv = ll(imag(outs[i])+.5);
		ll bv = ll(imag(outl[i])+.5) + ll(real(outs[i])+.5);
		res[i] = ((av % M * cut + bv) % M * cut + cv) % M;
	}
	return res;
}
// FFT mod end

// Fast Walsh-Hadamard Transform (FWHT) start
// Source: https://alan20210202.github.io/2020/08/07/FWHT/
// Modded xor FWHT function tested on AtCoder ABC212 H
//
// type: 1 = or, 2 = and, 3 = xor
// n is the smallest power of 2 at least sz(A)
// dir = 1/-1 for forward/inverse
// can change int *A to ll *A
void fwht(vector<ll> &A, int type, int dir = 1) {
	int n = 1;
	while(n < sz(A)) n <<= 1;
	A.resize(n, 0);
	for (int s = 2, h = 1; s <= n; s <<= 1, h <<= 1)
		for (int l = 0; l < n; l += s)
			for (int i = 0; i < h; i++) {
				if (type == 1) // or
					A[l + h + i] += dir * A[l + i];
				if (type == 2) // and
					A[l + i] += dir * A[l + h + i];
				if (type == 3) { // xor
					int t = A[l + h + i];
					A[l + h + i] = A[l + i] - t;
					A[l + i] = A[l + i] + t;
					if(dir < 0) A[l + h + i] /= 2, A[l + i] /= 2;
				}
			}
}
vector<ll> mult(vector<ll> &a, vector<ll> &b, int type)
{
	fwht(a,type);
	fwht(b,type);
	vector<ll> ans(sz(a));
	for(int i=0;i<sz(a);i++) ans[i]=a[i]*b[i];
	fwht(ans,type,-1);
	return ans;
}
// modded version
void fwht(vector<ll> &A, int type, int dir = 1) {
	const ll inv2 = inverse(2);
	int n = 1;
	while(n < sz(A)) n <<= 1;
	A.resize(n, 0);
	for (int s = 2, h = 1; s <= n; s <<= 1, h <<= 1)
		for (int l = 0; l < n; l += s)
			for (int i = 0; i < h; i++) {
				if (type == 1) // or
					A[l + h + i] = add(A[l + h + i], dir * A[l + i]);
				if (type == 2) // and
					A[l + i] = add(A[l + i], dir * A[l + h + i]);
				if (type == 3) { // xor
					int t = A[l + h + i];
					A[l + h + i] = add(A[l + i],  -t);
					A[l + i] = add(A[l + i], t);
					if(dir < 0) {
						A[l + h + i] = mult(A[l + h + i], inv2);
						A[l + i] = mult(A[l + i], inv2);
					}
				}
			}
}
vector<ll> mult(vector<ll> &a, vector<ll> &b, int type)
{
	fwht(a,type);
	fwht(b,type);
	vector<ll> ans(sz(a));
	for(int i=0;i<sz(a);i++) ans[i]=mult(a[i],b[i]);
	fwht(ans,type,-1);
	return ans;
}
// Fast Walsh-Hadamard Transform (FWHT) end

// Gauss elimination start
typedef vector<double> vd;
const double eps = 1e-12;
int solveLinear(vector<vd>& A, vd& b, vd& x) {
	int n = sz(A), m = sz(x), rank = 0, br, bc;
	if (n) assert(sz(A[0]) == m);
	vi col(m); iota(all(col), 0);
	forn(i,0,n) {
		double v, bv = 0;
		forn(r,i,n) forn(c,i,m)
		if ((v = fabs(A[r][c])) > bv)
			br = r, bc = c, bv = v;
		if (bv <= eps) {
			forn(j,i,n) if (fabs(b[j]) > eps) return -1;
			break;
		}
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		forn(j,0,n) swap(A[j][i], A[j][bc]);
		bv = 1/A[i][i];
		forn(j,i+1,n) {
			double fac = A[j][i] * bv;
			b[j] -= fac * b[i];
			forn(k,i+1,m) A[j][k] -= fac*A[i][k];
		}
		rank++;
	}
	x.assign(m, 0);
	for (int i = rank; i--;) {
		b[i] /= A[i][i];
		x[col[i]] = b[i];
		forn(j,0,i) b[j] -= A[j][i] * b[i];
		}
	return rank; // (multiple solutions i f rank < m)
}
// Gauss elimination end

// KACTL Geometry start
template <class T> int sgn(T x) { return (x > 0) - (x < 0); }
template<class T>
struct Point {
	typedef Point P;
	T x, y;
	explicit Point(T x=0, T y=0) : x(x), y(y) {}
	bool operator<(P p) const { return tie(x,y) < tie(p.x,p.y); }
	bool operator==(P p) const { return tie(x,y)==tie(p.x,p.y); }
	P operator+(P p) const { return P(x+p.x, y+p.y); }
	P operator-(P p) const { return P(x-p.x, y-p.y); }
	P operator*(T d) const { return P(x*d, y*d); }
	P operator/(T d) const { return P(x/d, y/d); }
	T dot(P p) const { return x*p.x + y*p.y; }
	T cross(P p) const { return x*p.y - y*p.x; }
	T cross(P a, P b) const { return (a-*this).cross(b-*this); }
	T dist2() const { return x*x + y*y; }
	double dist() const { return sqrt((double)dist2()); }
	// angle to x-axis in interval [-pi, pi]
	double angle() const { return atan2(y, x); }
	P unit() const { return *this/dist(); } // makes dist()=1
	P perp() const { return P(-y, x); } // rotates +90 degrees
	P normal() const { return perp().unit(); }
	// returns point rotated 'a' radians ccw around the origin
	P rotate(double a) const {
		return P(x*cos(a)-y*sin(a),x*sin(a)+y*cos(a)); }
	friend ostream& operator<<(ostream& os, P p) {
		return os << "(" << p.x << "," << p.y << ")"; }
};
template<class T> struct Point3D {
	typedef Point3D P;
	typedef const P& R;
	T x, y, z;
	explicit Point3D(T x=0, T y=0, T z=0) : x(x), y(y), z(z) {}
	bool operator<(R p) const {
		return tie(x, y, z) < tie(p.x, p.y, p.z); }
	bool operator==(R p) const {
		return tie(x, y, z) == tie(p.x, p.y, p.z); }
	P operator+(R p) const { return P(x+p.x, y+p.y, z+p.z); }
	P operator-(R p) const { return P(x-p.x, y-p.y, z-p.z); }
	P operator*(T d) const { return P(x*d, y*d, z*d); }
	P operator/(T d) const { return P(x/d, y/d, z/d); }
	T dot(R p) const { return x*p.x + y*p.y + z*p.z; }
	P cross(R p) const {
		return P(y*p.z - z*p.y, z*p.x - x*p.z, x*p.y - y*p.x);
	}
	T dist2() const { return x*x + y*y + z*z; }
	double dist() const { return sqrt((double)dist2()); }
	//Azimuthal angle (longitude) to x-axis in interval [-pi, pi]
	double phi() const { return atan2(y, x); }
	//Zenith angle (latitude) to the z-axis in interval [0, pi]
	double theta() const { return atan2(sqrt(x*x+y*y),z); }
	P unit() const { return *this/(T)dist(); } //makes dist()=1
	//returns unit vector normal to *this and p
	P normal(P p) const { return cross(p).unit(); }
	//returns point rotated 'angle' radians ccw around axis
	P rotate(double angle, P axis) const {
		double s = sin(angle), c = cos(angle); P u = axis.unit();
		return u*dot(u)*(1-c) + (*this)*c - cross(u)*s;
	}
};

// ------- Lines start -------
template<class P>
double lineDist(const P& a, const P& b, const P& p) {
	return (double)(b-a).cross(p-a)/(b-a).dist();
}
template<class P>
int sideOf(P s, P e, P p) { return sgn(s.cross(e, p)); }

template<class P>
int sideOf(const P& s, const P& e, const P& p, double eps) {
	auto a = (e-s).cross(p-s);
	double l = (e-s).dist()*eps;
	return (a > l) - (a < -l);
}
// {0/-1,{0,0}} if no inter/infinite inter, {1,point} otherwise
template<class P>
pair<int, P> lineInter(P s1, P e1, P s2, P e2) {
	auto d = (e1 - s1).cross(e2 - s2);
	if (d == 0) // if parallel
		return {-(s1.cross(e1, s2) == 0), P(0, 0)};
	auto p = s2.cross(e1, e2), q = s2.cross(e2, s1);
	return {1, (s1 * p + e1 * q) / d};
}
// projects p on line ab, refl returns reflection instead
template<class P>
P lineProj(P a, P b, P p, bool refl=false) {
	P v = b - a;
	return p - v.perp()*(1+refl)*v.cross(p-a)/v.dist2();
}
#define P Point<double>
P linearTransformation(const P& p0, const P& p1,
		const P& q0, const P& q1, const P& r) {
	P dp = p1-p0, dq = q1-q0, num(dp.cross(dq), dp.dot(dq));
	return q0 + P((r-p0).cross(num), (r-p0).dot(num))/dp.dist2();
}
#undef P
// ------- Lines end -------

// ------- Segments start -------
template<class P> bool onSegment(P s, P e, P p) {
	return p.cross(s, e) == 0 && (s - p).dot(e - p) <= 0;
}
#define P Point<double>
double segDist(P& s, P& e, P& p) {
	if (s==e) return (p-s).dist();
	auto d = (e-s).dist2(), t = min(d,max(.0,(p-s).dot(e-s)));
	return ((p-s)*d-(e-s)*t).dist()/d;
}
template<class P> vector<P> segInter(P a, P b, P c, P d) {
	auto oa = c.cross(d, a), ob = c.cross(d, b),
	     oc = a.cross(b, c), od = a.cross(b, d);
	// Checks if intersection is single non-endpoint point.
	if (sgn(oa) * sgn(ob) < 0 && sgn(oc) * sgn(od) < 0)
		return {(a * ob - b * oa) / (ob - oa)};
	set<P> s;
	if (onSegment(c, d, a)) s.insert(a);
	if (onSegment(c, d, b)) s.insert(b);
	if (onSegment(a, b, c)) s.insert(c);
	if (onSegment(a, b, d)) s.insert(d);
	return {all(s)};
}
#undef
// ------- Segments end -------

// ------- Polygons start -------
// The double of the area, points ccw order (cw gives negative)
template<class T>
T polygonArea2(vector<Point<T>>& v) {
	T a = v.back().cross(v[0]);
	rep(i,0,sz(v)-1) a += v[i].cross(v[i+1]);
	return a;
}
template<class P>
bool inPolygon(vector<P> &p, P a, bool strict = true) {
	int cnt = 0, n = sz(p);
	rep(i,0,n) {
		P q = p[(i + 1) % n];
		if (onSegment(p[i], q, a)) return !strict;
		//or: if (segDist(p[i], q, a) <= eps) return !strict;
		cnt ^= ((a.y<p[i].y) - (a.y<q.y)) * a.cross(p[i], q) > 0;
	}
	return cnt;
}
#define P Point<double>
P polygonCenter(const vector<P>& v) {
	P res(0, 0); double A = 0;
	for (int i = 0, j = sz(v) - 1; i < sz(v); j = i++) {
		res = res + (v[i] + v[j]) * v[j].cross(v[i]);
		A += v[j].cross(v[i]);
	}
	return res / A / 3;
}
vector<P> polygonCut(const vector<P>& poly, P s, P e) {
	vector<P> res;
	rep(i,0,sz(poly)) {
		P cur = poly[i], prev = i ? poly[i-1] : poly.back();
		bool side = s.cross(e, cur) < 0;
		if (side != (s.cross(e, prev) < 0))
			res.push_back(lineInter(s, e, cur, prev).second);
		if (side)
			res.push_back(cur);
	}
	return res;
}
double rat(P a, P b) { return sgn(b.x) ? a.x/b.x : a.y/b.y; }
double polyUnion(vector<vector<P>>& poly) {
	double ret = 0;
	rep(i,0,sz(poly)) rep(v,0,sz(poly[i])) {
		P A = poly[i][v], B = poly[i][(v + 1) % sz(poly[i])];
		vector<pair<double, int>> segs = {{0, 0}, {1, 0}};
		rep(j,0,sz(poly)) if (i != j) {
			rep(u,0,sz(poly[j])) {
				P C = poly[j][u], D = poly[j][(u + 1) % sz(poly[j])];
				int sc = sideOf(A, B, C), sd = sideOf(A, B, D);
				if (sc != sd) {
					double sa = C.cross(D, A), sb = C.cross(D, B);
					if (min(sc, sd) < 0)
						segs.emplace_back(sa / (sa - sb), sgn(sc - sd));
				} else if (!sc && !sd && j<i && sgn((B-A).dot(D-C))>0){
					segs.emplace_back(rat(C - A, B - A), 1);
					segs.emplace_back(rat(D - A, B - A), -1);
				}
			}
		}
		sort(all(segs));
		for (auto& s : segs) s.first = min(max(s.first, 0.0), 1.0);
		double sum = 0;
		int cnt = segs[0].second;
		rep(j,1,sz(segs)) {
			if (!cnt) sum += segs[j].first - segs[j - 1].first;
			cnt += segs[j].second;
		}
		ret += A.cross(B) * sum;
	}
	return ret / 2;
}
#undef P
template<class V, class L>
double signedPolyVolume(const V& p, const L& trilist) {
	double v = 0;
	for (auto i : trilist) v += p[i.a].cross(p[i.b]).dot(p[i.c]);
	return v / 6;
}
// ------- Polygons end -------

// ------- Hull start -------
#define P Point<ll>
vector<P> convexHull(vector<P> pts) {
	if (sz(pts) <= 1) return pts;
	sort(all(pts));
	vector<P> h(sz(pts)+1);
	int s = 0, t = 0;
	for (int it = 2; it--; s = --t, reverse(all(pts)))
		for (P p : pts) {
			while (t >= s + 2 && h[t-2].cross(h[t-1], p) <= 0) t--;
			h[t++] = p;
		}
	return {h.begin(), h.begin() + t - (t == 2 && h[0] == h[1])};
}
#undef
// points ccw order, strict = exclude boundary
bool inHull(const vector<Point<ll>>& l, Point<ll> p, bool strict = true) {
	int a = 1, b = sz(l) - 1, r = !strict;
	if (sz(l) < 3) return r && onSegment(l[0], l.back(), p);
	if (sideOf(l[0], l[a], l[b]) > 0) swap(a, b);
	if (sideOf(l[0], l[a], p) >= r || sideOf(l[0], l[b], p)<= -r)
		return false;
	while (abs(a - b) > 1) {
		int c = (a + b) / 2;
		(sideOf(l[0], l[c], p) > 0 ? b : a) = c;
	}
	return sgn(l[a].cross(l[b], p)) < r;
}

// - Line hull intersection start
#define cmp(i,j) sgn(dir.perp().cross(poly[(i)%n]-poly[(j)%n]))
#define extr(i) cmp(i + 1, i) >= 0 && cmp(i, i - 1 + n) < 0
template <class P> int extrVertex(vector<P>& poly, P dir) {
	int n = sz(poly), lo = 0, hi = n;
	if (extr(0)) return 0;
	while (lo + 1 < hi) {
		int m = (lo + hi) / 2;
		if (extr(m)) return m;
		int ls = cmp(lo + 1, lo), ms = cmp(m + 1, m);
		(ls < ms || (ls == ms && ls == cmp(lo, m)) ? hi : lo) = m;
	}
	return lo;
}
#define cmpL(i) sgn(a.cross(poly[i], b))
template <class P>
array<int, 2> lineHull(P a, P b, vector<P>& poly) {
	int endA = extrVertex(poly, (a - b).perp());
	int endB = extrVertex(poly, (b - a).perp());
	if (cmpL(endA) < 0 || cmpL(endB) > 0)
		return {-1, -1};
	array<int, 2> res;
	rep(i,0,2) {
		int lo = endB, hi = endA, n = sz(poly);
		while ((lo + 1) % n != hi) {
			int m = ((lo + hi + (lo < hi ? 0 : n)) / 2) % n;
			(cmpL(m) == cmpL(endB) ? lo : hi) = m;
		}
		res[i] = (lo + !cmpL(hi)) % n;
		swap(endA, endB);
	}
	if (res[0] == res[1]) return {res[0], -1};
	if (!cmpL(res[0]) && !cmpL(res[1]))
		switch ((res[0] - res[1] + sz(poly) + 1) % sz(poly)) {
			case 0: return {res[0], res[0]};
			case 2: return {res[1], res[1]};
		}
	return res;
}
// - Line hull intersection end

// Closest pair of points O(n log n)
#define P Point<ll>
pair<P, P> closest(vector<P> v) {
	assert(sz(v) > 1);
	set<P> S;
	sort(all(v), [](P a, P b) { return a.y < b.y; });
	pair<ll, pair<P, P>> ret{LLONG_MAX, {P(), P()}};
	int j = 0;
	for (P p : v) {
		P d{1 + (ll)sqrt(ret.first), 0};
		while (v[j].y <= p.y - d.x) S.erase(v[j++]);
		auto lo = S.lower_bound(p - d), hi = S.upper_bound(p + d);
		for (; lo != hi; ++lo)
			ret = min(ret, {(*lo - p).dist2(), {*lo, p}});
		S.insert(p);
	}
	return ret.second;
}
#undef P

// ------- Hull end-------

// KD-tree start
#define T long long
#define P Point<T>
const T INF = numeric_limits<T>::max();

bool on_x(const P& a, const P& b) { return a.x < b.x; }
bool on_y(const P& a, const P& b) { return a.y < b.y; }

struct Node {
	P pt; // if this is a leaf, the single point in it
	T x0 = INF, x1 = -INF, y0 = INF, y1 = -INF; // bounds
	Node *first = 0, *second = 0;

	T distance(const P& p) { // min squared distance to a point
		T x = (p.x < x0 ? x0 : p.x > x1 ? x1 : p.x);
		T y = (p.y < y0 ? y0 : p.y > y1 ? y1 : p.y);
		return (P(x,y) - p).dist2();
	}

	Node(vector<P>&& vp) : pt(vp[0]) {
		for (P p : vp) {
			x0 = min(x0, p.x); x1 = max(x1, p.x);
			y0 = min(y0, p.y); y1 = max(y1, p.y);
		}
		if (vp.size() > 1) {
			// split on x if width >= height (not ideal...)
			sort(all(vp), x1 - x0 >= y1 - y0 ? on_x : on_y);
			// divide by taking half the array for each child (not
			// best performance with many duplicates in the middle)
			int half = sz(vp)/2;
			first = new Node({vp.begin(), vp.begin() + half});
			second = new Node({vp.begin() + half, vp.end()});
		}
	}
};

struct KDTree {
	Node* root;
	KDTree(const vector<P>& vp) : root(new Node({all(vp)})) {}

	pair<T, P> search(Node *node, const P& p) {
		if (!node->first) {
			// uncomment if we should not find the point itself:
			// if (p == node->pt) return {INF, P()};
			return make_pair((p - node->pt).dist2(), node->pt);
		}

		Node *f = node->first, *s = node->second;
		T bfirst = f->distance(p), bsec = s->distance(p);
		if (bfirst > bsec) swap(bsec, bfirst), swap(f, s);

		// search closest side first, other side if needed
		auto best = search(f, p);
		if (bsec < best.first)
			best = min(best, search(s, p));
		return best;
	}

	// find nearest point to a point, and its squared distance
	// (requires an arbitrary operator< for Point)
	pair<T, P> nearest(const P& p) {
		return search(root, p);
	}
};
#undef T
#undef P
// KD-tree end

// KACTL Geometry end

// Geometry (self-made) start (incomplete)
template<class T> struct Point {
	T x,y;
	Point(): x(0), y(0) {}
	Point(T x, T y): x(x), y(y) {}
	Point operator+(const Point &p) { return {x+p.x, y+p.y}; }
	Point operator-(const Point &p) { return {x-p.x, y-p.y}; }
	Point operator*(T p) { return {x*p, y*p}; }
	Point operator/(T p) { return {x/p, y/p}; }
	Point translate(const Point &v) { return *this+v; }
	Point scale(const Point &c, ld factor) { return c+(*this-c)*factor; }
	Point rotate(double d) { return *this * polar(1.0, d); } // counter-clockwise, d in rad
	Point perp() { return {-y, x}; }
};
template<class T> bool operator==(const Point<T> &a, const Point<T> &b) { return a.x==b.x && a.y==b.y; }
template<class T> bool operator!=(const Point<T> &a, const Point<T> &b) { return !(a==b); }
template<class T> ostream& operator<<(ostream& out, const Point<T> &p) { return out<<"("<<p.x<<","<<p.y<<")"; }
template<class T> T sq(const Point<T> &p) { return p.x*p.x + p.y*p.y; }
template<class T> ld abs(const Point<T> &p) { return sqrtl(sq(p)); }
template<class T> T dot(const Point<T> &a, const Point<T> &b) { return a.x*b.x + a.y*b.y; }
template<class T> T cross(const Point<T> &a, const Point<T> &b) { return a.x*b.y - b.x*a.y; }
template<class T> bool isPerp(const Point<T> &a, const Point<T> &b) { return dot(a,b)==0; }
template<class T> T angle(const Point<T> &a, const Point<T> &b) {
	T cosTheta = dot(a,b) / abs(a) / abs(b);
	return acos(max(-1.0, min(1.0, cosTheta)));
}
template<class T> bool orient(const Point<T> &a, const Point<T> &b, const Point<T> &c) { return cross(b-a, c-a); }
template<class T> ld areaPoly(const vector<Point<T>> &v) {
	ld area = 0;
	for(int i=0,n=v.size();i<n;i++) area+=cross(v[i],v[(i+1)%n]);
	return abs(area)/2.0;
}

template<class T> struct Line {
	Point<T> v; T c;
	Line(const Point<T> &v, T c): v(v), c(c) {}
	Line(T a, T b, T c): v({b,-a}), c(c) {}    // ax+by=c
	Line(const Point<T> &a, const Point<T> &b): v(b-a), c(cross(v,a)) {} // between a and b
	bool operator()(const Point<T> &a, const Point<T> &b) { return dot(v,a)<dot(v,b); } // compare points on line in order
	T side(const Point<T> &p) { return cross(v,p)-c; }
	Line translate(Point<T> &p) { return {v, c+cross(v,p)}; }
	Line shiftLeft(double dist) { return {v, c+dist*abs(v)}; }
	Point<T> proj(Point<T> &p) { return p - perp(v)*side(p)/sq(v); }
	Point<T> reflect(Point<T> &p) { return p - perp(v)*2*side(p)/sq(v); }
};
template<class T> pair<int, Point<T>> lineInter(Line<T> &a, Line<T> &b) {
	T d = cross(a.v, b.v);
	if(d==0) return {0, {0,0}};
	Point<T> res = (b.v*a.c - a.v*b.c)*1.0L/d;
	return {1, res};
}
// Geometry (self-made) end

// Randomizer start
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int>(1,6)(rng);
uniform_int_distribution<> dis(1,6);

	Examples:
	cout<<rng()<<'\n';
	cout<<dis(rng)<<'\n';

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
// Randomizer end

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
const int dx[]={-1,0,1,-1,1,-1,0,1};
const int dy[]={-1,-1,-1,0,0,1,1,1};
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
