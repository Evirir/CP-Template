class PersistentSegmentTree
{
private:
    
    int index;
    int amount;
    vector<long long> tree;
    vector<pair<int, int>> child;
    vector<int> versions;
    
    int build(int l, int r, vector<long long>& a)
    {
        if (l > r) return -1;
        tree.push_back(0);
        child.emplace_back(-1, -1);
        int v = index++;
        if (l == r)
        {
            tree[v] = a[l];
            return v;
        }
        int m = (l + r) >> 1;
        int childV = build(l, m, a);
        child[v].first = childV;
        childV = build(m + 1, r, a);
        child[v].second = childV;
        tree[v] = (child[v].first == -1 ? 0 : tree[child[v].first]) + (child[v].second == -1 ? 0 : tree[child[v].second]);
        return v;
    }
    
    long long get(int a, int b, int v, int l, int r)
    {
        if (a > r || b < l || v == -1) return 0;
        if (a == l && b == r) return tree[v];
        int m = (l + r) >> 1;
        return get(a, min(b, m), child[v].first, l, m) + get(max(a, m + 1), b, child[v].second, m + 1, r);
    }
    
    int update(int pos, long long val, int v, int l, int r)
    {
        if (l > r || v == -1) return -1;
        tree.push_back(tree[v]);
        child.push_back(child[v]);
        int newV = index++;
        if (l == r)
        {
            tree[newV] += val;
            return newV;
        }
        int m = (l + r) >> 1;
        if (pos <= m)
        {
            int childV = update(pos, val, child[v].first, l, m);
            child[newV].first = childV;
        }
        else
        {
            int childV = update(pos, val, child[v].second, m + 1, r);
            child[newV].second = childV;
        }
        tree[newV] = (child[newV].first == -1 ? 0 : tree[child[newV].first]) + (child[newV].second == -1 ? 0 : tree[child[newV].second]);
        return newV;
    }

public:
    
    PersistentSegmentTree(int n) : amount(n), index(0)
    {
        vector<long long> a(n, 0);
        versions.push_back(build(0, n - 1, a));
    }
    
    PersistentSegmentTree(int n, vector<long long>& a) : amount(n), index(0)
    {
        versions.push_back(build(0, n - 1, a));
    }
    
    long long get(int version, int l, int r)
    {
        return get(l, r, versions[version], 0, amount - 1);
    }
    
    void update(int version, int pos, long long val)
    {
        versions.push_back(update(pos, val, versions[version], 0, amount - 1));
    }
    
    int lastVersion()
    {
        return (int)versions.size() - 1;
    }
};