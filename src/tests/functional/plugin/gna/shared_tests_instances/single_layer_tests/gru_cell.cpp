// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gru_cell.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

class GRUCellGNATest : public GRUCellTest {
std::vector<std::string> m_activations = {};
size_t m_input_size = 0;

protected:
    void SetUp() override {
        GRUCellTest::SetUp();
        std::map<std::string, std::string> additionalConfig;
        std::tie(std::ignore, std::ignore, std::ignore, m_input_size, m_activations, std::ignore, std::ignore,
            std::ignore, std::ignore, additionalConfig) = this->GetParam();

        for (const auto configEntry : additionalConfig) {
            if ("GNA_SW_EXACT" == configEntry.second)
                threshold = 0.1f;
        }
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        auto max = 0.035f;
        auto min = -max;
        std::vector<std::string> tanhReluAct = {"tanh", "relu"};
        if (tanhReluAct == m_activations) {
            if (m_input_size == 1) {
                max = 0.055f;
            }
            min = -0.1f;
        }
        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), min, max);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }
};

TEST_P(GRUCellGNATest, CompareWithRefs) {
        Run();
}

}  //  namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{1};
    std::vector<size_t> hidden_size{1, 5};
    std::vector<size_t> input_size{1, 10};
    std::vector<std::vector<std::string>> activations = {{"relu", "tanh"}, {"tanh", "sigmoid"}, {"sigmoid", "tanh"},
                                                         {"tanh", "relu"}};
    std::vector<float> clip = {0.0f, 0.7f};
    std::vector<bool> linear_before_reset = {true, false};
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    std::vector<std::map<std::string, std::string>> additional_config = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        },
        {
            {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
        }
    };

    INSTANTIATE_TEST_SUITE_P(smoke_GRUCellCommon, GRUCellGNATest,
            ::testing::Combine(
            ::testing::ValuesIn(should_decompose),
            ::testing::ValuesIn(batch),
            ::testing::ValuesIn(hidden_size),
            ::testing::ValuesIn(input_size),
            ::testing::ValuesIn(activations),
            ::testing::ValuesIn(clip),
            ::testing::ValuesIn(linear_before_reset),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(additional_config)),
            GRUCellTest::getTestCaseName);

}  // namespace
