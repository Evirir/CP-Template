# CP-Template
C++ Templates for Competitive Programming

Main file is Data Structures.cpp

See also: [zscoder's template](https://github.com/zscoder/CompetitiveProgramming/blob/master/Data%20Structures%20Class%20Template.cpp)

List of things (in order):

1. Segment/Fenwick tree
	- Segment tree (all ranges are closed i.e. l,r inclusive)
		- Point update
		- Range update (Lazy propagation)
		- Short iterative ver. [[source](https://codeforces.com/blog/entry/18051)]
		- 2D segment tree
		- Persistent segment tree
		- Segment tree beats (e.g. range min/max update) [[source](https://tjkendev.github.io/procon-library/cpp/range_query/segment_tree_beats_2.html)]
	- Fenwick tree: point and range update

2. String algorithms
	- Prefix function (KMP)
	- Z-algorithm
	- Trie
	- Aho-Corasick
	- Suffix Array

3. Graph theory
	- Algorithms/DS
		- DSU (Disjoint-set union)
		- Kruskal
		- Dijkstra
		- Floyd-Warshall
		- SPFA (Shortest path faster algorithm)/Bellman-Ford
		- Dinic Flow O(V^2E)
		- Edmonds-Karp: Min Cost Max Flow
		- Hopcroft-Karp matching (max-cardinality bipartite matching/MCBM)
		- Strongly connected component (SCC): Tarjan's algorithm
	- Common Techniques
		- Euler tour compression
		- Heavy-light decomposition (HLD)
		- Lowest Common Ancestor (LCA)
			- Euler tour method: O(log n) query
			- Depth method: O(log n) query
			- Sparse table: O(1) query but long
		- Centroid decomposition: solving for all paths crossing current centroid 

4. Data structures
	- Sparse table
	- Convex hull trick (CHT)
		- Dynamic version (LineContainer): O(log n) query
		- Offline version: O(1) query
	- Li Chao Tree \[untested\]

5. Maths
	- Combinatorics
		- Modular operations: Add, mult, inverse, binary exponentiation, binomial coefficients, factorials
		- getpf(): O(sqrt(n)) prime factorization
	- Matrix exponentiation
	- Number theory
	
6. Square root decomposition/Mo's algorithm

7. Convolutions
	- Fast Fourier Transform (FFT)
	- Number Theoretic Transform (NTT)
	- FFT Mod

8. Geometry \[untested\]

9. Miscellaneous
	- Randomizer (Mersenne prime twister, mt19937)
	- unordered_map/hash map custom hash (http://xorshift.di.unimi.it/splitmix64.c)
	- Binary converter: print numbers in binary
	- Grid movement: 4/8 directions
	- Nearest pair of points
