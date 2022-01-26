// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/embedding_bag_packed_sum.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(EmbeddingBagPackedSumLayerTest, Serialize) {
        Serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U8};

    const std::vector<InferenceEngine::Precision> indPrecisions = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I32};

    const std::vector<std::vector<size_t>> emb_table_shape = {{5, 6}, {10, 35}, {5, 4, 16}};
    const std::vector<std::vector<std::vector<size_t>>> indices =
        {{{0, 1}, {2, 2}}, {{4, 4, 3}, {1, 0, 2}}, {{1, 2, 1, 2}}};
    const std::vector<bool> with_weights = {false, true};

    const auto EmbeddingBagPackedSumParams = ::testing::Combine(
        ::testing::ValuesIn(emb_table_shape),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(with_weights));

    INSTANTIATE_TEST_SUITE_P(
        smoke_EmbeddingBagPackedSumLayerTest_Serialization, EmbeddingBagPackedSumLayerTest,
        ::testing::Combine(EmbeddingBagPackedSumParams,
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(indPrecisions),
                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        EmbeddingBagPackedSumLayerTest::getTestCaseName);
} // namespace
