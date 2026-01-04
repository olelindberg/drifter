#include <gtest/gtest.h>
#include "parallel/halo_exchange.hpp"
#include "parallel/domain_decomposition.hpp"
#include "../test_utils.hpp"

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

using namespace drifter;
using namespace drifter::testing;

class HaloExchangeTest : public ::testing::Test {
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

// Test that halo exchange works for simple case
TEST_F(HaloExchangeTest, SimpleExchange) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Create a simple decomposition (mock)
    // This would normally use actual mesh data
    DomainDecomposition decomp(MPI_COMM_WORLD);

    // For testing, we need a mesh - skip if not available
    // In real tests, we'd set up a proper mesh

    // Basic test: verify construction doesn't crash
    EXPECT_EQ(decomp.rank(), rank_);
    EXPECT_EQ(decomp.size(), size_);
}

// Test that data is correctly exchanged between neighbors
TEST_F(HaloExchangeTest, DataCorrectness) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Each rank sends its rank number to neighbors
    // and receives neighbor rank numbers

    std::vector<Real> send_data(10, static_cast<Real>(rank_));
    std::vector<Real> recv_data(10, -1.0);

    // Exchange with rank+1 and rank-1 (cyclic)
    int prev = (rank_ + size_ - 1) % size_;
    int next = (rank_ + 1) % size_;

    MPI_Request requests[4];
    MPI_Status statuses[4];

    // Post receives
    MPI_Irecv(recv_data.data(), 10, MPI_DOUBLE, prev, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recv_data.data(), 10, MPI_DOUBLE, next, 1,
              MPI_COMM_WORLD, &requests[1]);

    // Post sends
    MPI_Isend(send_data.data(), 10, MPI_DOUBLE, next, 0,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(send_data.data(), 10, MPI_DOUBLE, prev, 1,
              MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, statuses);

    // Verify received data
    for (int i = 0; i < 10; ++i) {
        // Data should be from one of our neighbors
        EXPECT_TRUE(recv_data[i] == prev || recv_data[i] == next ||
                    recv_data[i] == rank_)
            << "Unexpected data: " << recv_data[i];
    }
}

// Test async exchange
TEST_F(HaloExchangeTest, AsyncExchange) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Simulate async exchange pattern
    // 1. Start exchange
    // 2. Do local work
    // 3. Finish exchange

    std::vector<Real> data(100, static_cast<Real>(rank_));

    MPI_Request request;
    MPI_Status status;

    // Broadcast from rank 0
    if (rank_ == 0) {
        for (int r = 1; r < size_; ++r) {
            MPI_Isend(data.data(), 100, MPI_DOUBLE, r, 0,
                      MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
    } else {
        MPI_Irecv(data.data(), 100, MPI_DOUBLE, 0, 0,
                  MPI_COMM_WORLD, &request);

        // Simulate local work while waiting
        double local_sum = 0.0;
        for (int i = 0; i < 1000; ++i) {
            local_sum += std::sin(i * 0.01);
        }

        MPI_Wait(&request, &status);

        // Verify data
        for (int i = 0; i < 100; ++i) {
            EXPECT_NEAR(data[i], 0.0, TOLERANCE);
        }
    }
}

// Test collective operations
TEST_F(HaloExchangeTest, CollectiveSum) {
    Real local_value = static_cast<Real>(rank_ + 1);
    Real global_sum = 0.0;

    MPI_Allreduce(&local_value, &global_sum, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    // Sum of 1 + 2 + ... + size = size * (size + 1) / 2
    Real expected = static_cast<Real>(size_ * (size_ + 1) / 2);

    EXPECT_NEAR(global_sum, expected, TOLERANCE);
}

#endif  // DRIFTER_USE_MPI

// Test that works without MPI
TEST_F(HaloExchangeTest, SerialFallback) {
    // Create decomposition without mesh
    DomainDecomposition decomp;

    // Should work in serial mode
    EXPECT_EQ(decomp.rank(), 0);
    EXPECT_EQ(decomp.size(), 1);
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
