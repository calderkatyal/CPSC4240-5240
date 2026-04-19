#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include <iomanip>
// Include ParlayLib (adjust the path if needed)
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"

// A simple 2D point structure
struct Point2D {
  double x, y;
  Point2D(double xx=0.0, double yy=0.0) : x(xx), y(yy) {}
};

// A helper to compute squared distance
inline double squared_distance(const Point2D& a, const Point2D& b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return dx*dx + dy*dy;
}

// KD-Tree node
struct KDNode {
  int axis;          // 0 for x, 1 for y
  double splitValue; // coordinate pivot
  int pointIndex;    // index in the original array

  KDNode* left;
  KDNode* right;

  KDNode() : axis(0), splitValue(0.0), pointIndex(-1),
             left(nullptr), right(nullptr) {}
};

// DistIndex for storing (distance^2, index)
struct DistIndex {
  double dist;
  int index;
  DistIndex(double d=0, int i=0) : dist(d), index(i) {}
};

// For a max-heap, we want to put the largest distance on top
inline bool operator<(const DistIndex &a, const DistIndex &b) {
  return a.dist < b.dist;
}

// TODO: Implement a function to build the kd-tree
KDNode* build_kd_tree(
    parlay::slice<int*, int*> indices,
    const parlay::sequence<Point2D>& points,
    int depth = 0
) {
  // 1) Base cases: if 0 or 1 points, create a leaf node or return
  size_t n = indices.size();
  if (n == 0) return nullptr;

  if (n == 1) {
    KDNode* leaf = new KDNode();
    leaf->pointIndex = indices[0];
    leaf->axis = depth % 2;
    leaf->splitValue = (leaf->axis == 0) ? points[indices[0]].x : points[indices[0]].y;
    return leaf;
  }

  // 2) Determine axis = (depth % 2)
  int axis = depth % 2;

  // 3) Sort indices by that axis (x or y)
  // Use std::nth_element for O(n) median selection — faster than full sort
  size_t mid = n / 2;
  int* base = indices.begin();
  if (axis == 0) {
    std::nth_element(base, base + mid, base + n,
      [&](int a, int b) { return points[a].x < points[b].x; });
  } else {
    std::nth_element(base, base + mid, base + n,
      [&](int a, int b) { return points[a].y < points[b].y; });
  }

  // 4) Find median index
  int medianIdx = base[mid];

  // 5) Create a node with that pivot
  KDNode* node = new KDNode();
  node->axis = axis;
  node->splitValue = (axis == 0) ? points[medianIdx].x : points[medianIdx].y;
  node->pointIndex = medianIdx;

  // 6) Recurse left and right in parallel
  auto leftSlice  = indices.cut(0, mid);
  auto rightSlice = indices.cut(mid + 1, n);

  parlay::par_do(
      [&]() { node->left  = build_kd_tree(leftSlice,  points, depth + 1); },
      [&]() { node->right = build_kd_tree(rightSlice, points, depth + 1); }
    );

  return node;
}
// KNN Helper: holds a local max-heap of size k
class KNNHelper {
public:
  KNNHelper(const parlay::sequence<Point2D>& pts, int kk)
    : points(pts), k(kk) {
    best.reserve(k);
  }

  // Perform recursive search
  void search(const KDNode* node, const Point2D& q) {
    // TODO:
    if (node == nullptr) return;

    // 1) compute dist2 from q to node->pointIndex
    double dist2 = squared_distance(q, points[node->pointIndex]);

    // 2) update_best if needed
    update_best(dist2, node->pointIndex);

    // 3) compare q's coordinate to splitValue
    double diff = (node->axis == 0) ? (q.x - node->splitValue) : (q.y - node->splitValue);
    double diff2 = diff * diff;

    // 4) search near side, possibly search far side if needed
    // Near side = side q falls on; far side is other
    const KDNode* near_child = (diff <= 0.0) ? node->left : node->right;
    const KDNode* far_child = (diff <= 0.0) ? node->right : node->left;

    search(near_child, q);

    // We will only search far side if splitting plane closer than current worst,
    // or if we haven't found k neighbors yet
    if ((int)best.size() < k || diff2 < best[0].dist) {
      search(far_child, q);
    }
  }
  // Return final results sorted by ascending distance
  parlay::sequence<DistIndex> get_results() const {
    parlay::sequence<DistIndex> result(best.begin(), best.end());
    parlay::sort_inplace(result, [&](auto &a, auto &b){
      return a.dist < b.dist;
    });
    return result;
  }

private:
  const parlay::sequence<Point2D>& points;
  int k;
  std::vector<DistIndex> best; // will be a max-heap

  // If we have fewer than k, push. Otherwise compare with largest so far.
  void update_best(double dist2, int idx) {
    // TODO: use a max-heap for best and update best with dist2 and idx
    if ((int)best.size() < k) {
      best.emplace_back(dist2, idx);
      // Re-heapify
      std::push_heap(best.begin(), best.end());
    } else if (dist2 < best[0].dist) {
      // Replace worst (largest) element
      std::pop_heap(best.begin(), best.end());
      best.back() = DistIndex(dist2, idx);
      std::push_heap(best.begin(), best.end());
    }
  }
};

// Parallel k-NN for all queries
parlay::sequence<parlay::sequence<DistIndex>>
knn_search_all(const KDNode* root,
               const parlay::sequence<Point2D>& data_points,
               const parlay::sequence<Point2D>& query_points,
               int k) {
  int Q = (int)query_points.size();
  parlay::sequence<parlay::sequence<DistIndex>> results(Q);

  parlay::parallel_for(0, Q, [&](int i){
    KNNHelper helper(data_points, k);
    helper.search(root, query_points[i]);
    results[i] = helper.get_results();
  });

  return results;
}

// A function to load points from a file
parlay::sequence<Point2D> load_points_from_file(const std::string &filename) {
  // TODO: open file, read N, read N lines of x y into a parlay::sequence
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    std::cerr << "ERROR: cannot open " << filename << "\n";
    return {};
  }

  int N;
  fin >> N;
  parlay::sequence<Point2D> pts(N);
  for (int i = 0; i < N; i++) {
    fin >> pts[i].x >> pts[i].y;
  }
  fin.close();
  return pts;
}
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <data_file> <query_file> <k>\n";
    return 1;
  }

  std::string data_file  = argv[1];
  std::string query_file = argv[2];
  int k = std::stoi(argv[3]);

  auto data_points = load_points_from_file(data_file);
  int N = (int)data_points.size();

  parlay::sequence<int> indices(N);
  parlay::parallel_for(0, N, [&](int i){ indices[i] = i; });
  KDNode* root = build_kd_tree(indices.cut(0, N), data_points, 0);

  auto query_points = load_points_from_file(query_file);
  int Q = (int)query_points.size();

  auto results = knn_search_all(root, data_points, query_points, k);

  for (int q = 0; q < Q; q++) {
    std::cout << "Query " << q << " : ("
              << query_points[q].x << ", "
              << query_points[q].y << ")\n";
    std::cout << "  kNN: ";
    for (auto &di : results[q]) {
      std::cout << "(dist2=" << di.dist
                << ", idx=" << di.index << ") ";
    }
    std::cout << "\n";
  }

  return 0;
}
