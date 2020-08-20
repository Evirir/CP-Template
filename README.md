# CP-Template
C++ Templates for Competitive Programming

List of things (in order):

1. Segment/Fenwick tree
	- Segment tree (all ranges are closed i.e. l,r inclusive)
		- Point update
		- Range update (Lazy propoagation)
		- Short iterative ver on CF
	- Fenwick tree: point and range update
	- Segment tree beats (e.g. range min/max update)

2. String algorithms
	- Prefix function (KMP)
	- Z-algorithm
	- Trie

3. Graph theory
	- Classical
		- DSU (Disjoint-set union)
		- Kruskal
		- Dijkstra
		- Floyd-Warshall
		- SPFA (Shortest path faster algorithm)/Bellman-Ford
		- Dinic Flow O(V^2E)
		- Edmond's Karp: Min Cost Max Flow
		- Hopkroft-Karp matching (max-cardinality bipartite matching/MCBM)
		- Strongly connected component (SCC): Tarjan's algorithm
	- Problem dependent
		- Euler tour compression
		- Heavy-light decomposition (HLD)
		- Lowest Common Ancestor (LCA)
			- Euler tour method: O(log n) query
			- Depth method: O(log n) query
			- Sparse table: O(1) query but long
		- Centroid decomposition: solving for all paths crossing current centroid 

4. Data structures
	- Sparse table
	- Convex hull trick
		- Dynamic version: O(log n) query
		- Everything is sorted version: O(1) query

5. Maths
	- Combinatorics
		- Modular operations: Add, mult, inverse, binary exponentiation, binomial coefficients, factorials
		- getpf(): O(sqrt(n)) prime factorization
	- Matrix exponentiation
	- Number theory
	
6. Square root decomposition/Mo's algorithm

7. Miscellaneous
	- Randomizer (Mersenne prime twister, mt19937)
	- Binary converter: print numbers in binary
	- Grid movement: 4/8 directions
	- Nearest pair of points
