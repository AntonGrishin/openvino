// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "constant.hpp"
#include "dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class Convolution {
public:
    Convolution();

    Convolution(
        const DequantizationOperations::Subtract zeroPointOnActivations,
        const Constant& constantOnWeights,
        const DequantizationOperations& dequantizationOnWeights);

    bool empty() const;

    DequantizationOperations::Subtract zeroPointOnActivations;
    Constant constantOnWeights;
    DequantizationOperations dequantizationOnWeights;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
