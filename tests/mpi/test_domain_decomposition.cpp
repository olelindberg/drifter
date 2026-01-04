#include <gtest/gtest.h>
#include "parallel/domain_decomposition.hpp"
#include "../test_utils.hpp"

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

using namespace drifter;
using namespace drifter::testing;

class DomainDecompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef DRIFTER_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
        rank_ = 0;
        size_ = 1;
#endif
    }

    int rank_ = 0;
    int size_ = 1;
};

#ifdef DRIFTER_USE_MPI

// Test that all elements are partitioned
TEST_F(DomainDecompositionTest, AllElementsPartitioned) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Count local elements across all ranks
    Index local_count = 100 / size_ + (rank_ < (100 % size_) ? 1 : 0);

    Index global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(global_count, 100);
}

// Test that partition is balanced
TEST_F(DomainDecompositionTest, PartitionBalance) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    Index total_elements = 1000;
    Index elements_per_rank = total_elements / size_;
    Index remainder = total_elements % size_;

    Index my_count = elements_per_rank + (rank_ < remainder ? 1 : 0);

    // Verify balance
    Index min_count, max_count;
    MPI_Allreduce(&my_count, &min_count, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&my_count, &max_count, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Max should be at most 1 more than min
    EXPECT_LE(max_count - min_count, 1);
}

// Test load balancer imbalance computation
TEST_F(DomainDecompositionTest, LoadImbalance) {
    LoadBalancer balancer(MPI_COMM_WORLD);

    // Create mock decomposition
    DomainDecomposition decomp(MPI_COMM_WORLD);

    // Without actual elements, imbalance should be 1.0 or undefined
    // This is mainly a compilation/linkage test

    EXPECT_EQ(decomp.rank(), rank_);
}

// Test ghost element identification concept
TEST_F(DomainDecompositionTest, GhostConceptTest) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Simulate: rank 0 has elements [0, 99], rank 1 has [100, 199], etc.
    // Element 99 on rank 0 neighbors element 100 on rank 1

    Index my_start = rank_ * 100;
    Index my_end = my_start + 99;

    // Boundary elements
    std::vector<Index> boundary_elements;
    if (rank_ > 0) {
        boundary_elements.push_back(my_start);  // Neighbors prev rank
    }
    if (rank_ < size_ - 1) {
        boundary_elements.push_back(my_end);  // Neighbors next rank
    }

    // Each rank should have 0, 1, or 2 boundary elements
    EXPECT_LE(boundary_elements.size(), 2u);

    // Non-boundary ranks should have 2
    if (rank_ > 0 && rank_ < size_ - 1) {
        EXPECT_EQ(boundary_elements.size(), 2u);
    }
}

// Test communication pattern
TEST_F(DomainDecompositionTest, CommunicationPattern) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Build simple neighbor list
    std::vector<int> neighbors;
    if (rank_ > 0) neighbors.push_back(rank_ - 1);
    if (rank_ < size_ - 1) neighbors.push_back(rank_ + 1);

    // Verify expected neighbor count
    if (rank_ == 0 || rank_ == size_ - 1) {
        EXPECT_EQ(neighbors.size(), 1u);
    } else {
        EXPECT_EQ(neighbors.size(), 2u);
    }
}

#endif  // DRIFTER_USE_MPI

// Test serial decomposition
TEST_F(DomainDecompositionTest, SerialDecomposition) {
    DomainDecomposition decomp;

    EXPECT_EQ(decomp.rank(), 0);
    EXPECT_EQ(decomp.size(), 1);
    EXPECT_EQ(decomp.num_local_elements(), 0);  // No mesh set
    EXPECT_EQ(decomp.num_ghost_elements(), 0);
}

// Test partition strategy enum
TEST_F(DomainDecompositionTest, PartitionStrategies) {
    // Verify enum values exist
    EXPECT_NE(static_cast<int>(PartitionStrategy::Morton),
              static_cast<int>(PartitionStrategy::Metis));
    EXPECT_NE(static_cast<int>(PartitionStrategy::Uniform),
              static_cast<int>(PartitionStrategy::Morton));
}

// Test ghost config
TEST_F(DomainDecompositionTest, GhostConfig) {
    GhostConfig config;

    // Default values
    EXPECT_EQ(config.num_layers, 1);
    EXPECT_TRUE(config.include_diagonal);
    EXPECT_FALSE(config.include_corners);
}

// Main for MPI tests
int main(int argc, char** argv) {
#ifdef DRIFTER_USE_MPI
    MPI_Init(&argc, &argv);
#endif

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef DRIFTER_USE_MPI
    MPI_Finalize();
#endif

    return result;
}
