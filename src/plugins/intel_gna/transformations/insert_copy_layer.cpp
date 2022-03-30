// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_copy_layer.hpp"
#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ops/copy.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ie/ie_common.h>
#include <openvino/core/except.hpp>

#include "gna_plugin_log.hpp"
#include "gna_lib_ver_selector.hpp"
#include "layers/gna_permute.hpp"

namespace GNAPluginNS {

NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcat, "HandleMultiConnectedLayerToConcat", 0);
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeMemoryLayer, "InsertCopyBeforeMemoryLayer", 0);
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeConcatLayer, "InsertCopyBeforeConcatLayer", 0);
NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcatAndMemory, "HandleMultiConnectedLayerToConcatAndMemory", 0);
NGRAPH_RTTI_DEFINITION(MatchNonFunctionalLayers, "MatchNonFunctionalLayers", 0);
NGRAPH_RTTI_DEFINITION(HandleNonComputationalSubgraphs, "HandleNonComputationalSubgraphs", 0);


namespace {
    void InsertCopyLayerBetween(std::shared_ptr<ngraph::Node> input_op,
                                std::shared_ptr<ngraph::Node> output_op,
                                const size_t& index) {
        NGRAPH_CHECK(input_op);
        NGRAPH_CHECK(output_op);

        auto copy_op = std::make_shared<GNAPluginNS::Copy>(input_op->output(output_op->input(index).get_source_output().get_index()));
        copy_op->set_friendly_name(input_op->get_friendly_name() + "/copy_layer/" + output_op->get_friendly_name() + "." + std::to_string(index));
        ngraph::copy_runtime_info(input_op, copy_op);

        output_op->input(index).replace_source_output(copy_op);
    }

    bool IsCropAffined(std::shared_ptr<ngraph::Node> node) {
        auto crop = std::dynamic_pointer_cast<ngraph::op::CropIE>(node);
        if (crop != nullptr && !crop->offset.empty()) {
            // currently crop layer only supports 2 bytes in int16 and int8 mode.
            // In fp32 mode this is not necessary but is useful for testing
            size_t bytesPerCropElement = 2;
            size_t cropOffset = crop->offset.back() * bytesPerCropElement;
            return (ALIGN64(cropOffset) != cropOffset);
        }
        return false;
    }

    // this not only mathematically trivial, has some WA for kaldi case
    bool IsTrivialPermute(std::shared_ptr<ngraph::Node> node) {
        auto permute = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(node);
        if (!permute) return false;

        auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(permute->input_value(1).get_node_shared_ptr());
        if (!transpose_const) return false;

        auto node_order = transpose_const->cast_vector<int64_t>();

        if (node_order == std::vector<int64_t>({ 0, 3, 2, 1 })) {
            return true;  // supported case
        }

        if (permute->get_input_size() == 0)
            return false; // unsupported case

        auto input = permute->input(0).get_source_output().get_node_shared_ptr();
        auto input_order = permute->get_input_shape(0);
        // auto inputs = layer->insData.begin()->lock();
        // auto inputsOrder = inputs->getTensorDesc().getDims();
        // cases when all permutations happened either between 1 and X shape where no other dims in between
        auto permute_seq = genPermutations(node_order.begin(), node_order.end());
        auto input_order_transformed = input_order;
        for (auto && permute : permute_seq) {
            // check dims of permuted
            if (input_order_transformed[permute.first] == 1 &&
                input_order_transformed[permute.second] == 1) {
                return true;
            }
            if (input_order_transformed[permute.first] != 1 &&
                input_order_transformed[permute.second] != 1) {
                return false;
            }
            // check dims in between
            for (int j = std::min(permute.first, permute.second) + 1; j < std::max(permute.first, permute.second); j++) {
                if (input_order_transformed[j] != 1) {
                    return false;
                }
            }
            // apply permutation
            std::swap(input_order_transformed[permute.first], input_order_transformed[permute.second]);
        }
        return true;
    }

    bool IsNonFunctionalGNANode(std::shared_ptr<ngraph::Node> node) {
        return std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Squeeze>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(node) ||
               IsTrivialPermute(node);
    }
}// namespace

HandleMultiConnectedLayerToConcat::HandleMultiConnectedLayerToConcat() {
    MATCHER_SCOPE(HandleMultiConnectedLayerToConcat);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(m.get_match_root());
        if (!concat) return false;

        std::set<std::shared_ptr<ngraph::Node>> inputs;
        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto input_op = concat->get_input_node_shared_ptr(i);

            if (inputs.find(input_op) != inputs.end()) {
                InsertCopyLayerBetween(input_op, concat, i);
            } else {
                inputs.insert(input_op);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_op, matcher_name);
    this->register_matcher(m, callback);
}


InsertCopyBeforeMemoryLayer::InsertCopyBeforeMemoryLayer() {
    MATCHER_SCOPE(InsertCopyBeforeMemoryLayer);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat,
                                                ngraph::op::ReadValueBase,
                                                ngraph::op::AssignBase>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());

        if (!std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(node) &&
            !std::dynamic_pointer_cast<ngraph::op::AssignBase>(node))
            return false;

        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto current_node = node->get_input_node_shared_ptr(i);
            auto matched_node_input = current_node;

            while (IsNonFunctionalGNANode(current_node)) {
                current_node = current_node->get_input_node_shared_ptr(0);
            }

            // // Crop -> Memory, Input -> Split -> Memory, Concat -> Memory
            if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) && !IsCropAffined(current_node)) ||
                std::dynamic_pointer_cast<ngraph::opset8::Concat>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(current_node)) {
                    InsertCopyLayerBetween(matched_node_input, node, i);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_op, matcher_name);
    this->register_matcher(m, callback);
}

InsertCopyBeforeConcatLayer::InsertCopyBeforeConcatLayer() {
    MATCHER_SCOPE(InsertCopyBeforeConcatLayer);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat,
                                                ngraph::op::ReadValueBase,
                                                ngraph::op::AssignBase>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(m.get_match_root());
        if (!concat)
            return false;

        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto current_node = concat->get_input_node_shared_ptr(i);
            auto concat_input = current_node;

            while (IsNonFunctionalGNANode(current_node)) {
                current_node = current_node->get_input_node_shared_ptr(0);
            }

            // // Crop -> Concat, Input -> Split -> Concat
            if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) && !IsCropAffined(current_node)) ||
                std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(current_node)) {
                    InsertCopyLayerBetween(concat_input, concat, i);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_op, matcher_name);
    this->register_matcher(m, callback);
}

bool HandleMultiConnectedLayerToConcatAndMemory::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleMultiConnectedLayerToConcatAndMemory);

    bool is_graph_modified = false;
    for (auto& node : f->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset8::Constant>(node) || IsNonFunctionalGNANode(node))
            continue;
        for (auto& output : node->outputs()) {
            auto inputTo = output.get_target_inputs();
            if (inputTo.size() < 2) continue;
            std::vector<std::pair<std::shared_ptr<ngraph::Node>, size_t>> concat_nodes, memory_nodes;
            for (auto& child : inputTo) {
                auto current_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto copy_output_node = current_node;
                auto previous_node = node;
                auto current_index = child.get_index();

                while ((IsNonFunctionalGNANode(current_node))) {
                    if (current_node->get_output_size() == 0) break;
                    if (current_node->output(0).get_target_inputs().size()  == 0) break;
                    previous_node = current_node;
                    current_node = current_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                }

                if (std::dynamic_pointer_cast<ngraph::opset8::Concat>(current_node)) {
                    concat_nodes.push_back(std::make_pair(copy_output_node, current_index));
                } else if (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(current_node) ||
                    std::dynamic_pointer_cast<ngraph::op::AssignBase>(current_node)) {
                    memory_nodes.push_back(std::make_pair(copy_output_node, current_index));
                }
            }

            if (memory_nodes.empty() && concat_nodes.empty()) continue;
            auto count_to_copy = memory_nodes.size() + concat_nodes.size() - (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ? 0 : 1);
            // Insertion of copy to memory layers have a priority on the concat layers
            for (size_t i = 0; i < count_to_copy; i++) {
                auto out_layer = (i < memory_nodes.size()) ? memory_nodes[i].first : concat_nodes[i - memory_nodes.size()].first;
                auto input_id = (i < memory_nodes.size()) ? memory_nodes[i].second : concat_nodes[i - memory_nodes.size()].second;
                InsertCopyLayerBetween(node, out_layer, input_id);
            }
            is_graph_modified = true;
        }
    }

    return is_graph_modified;
}

MatchNonFunctionalLayers::MatchNonFunctionalLayers() {
    MATCHER_SCOPE(MatchNonFunctionalLayers);

    auto noncompute_op = ngraph::pattern::wrap_type<ngraph::opset8::Reshape,
                                                ngraph::opset8::Squeeze,
                                                ngraph::opset8::Unsqueeze,
                                                ngraph::opset8::Transpose,
                                                ngraph::op::CropIE,
                                                ngraph::opset8::Split,
                                                ngraph::opset8::VariadicSplit,
                                                ngraph::opset8::Parameter,
                                                ngraph::opset8::Constant,
                                                ngraph::opset8::Result>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());
        if (!(IsNonFunctionalGNANode(node) ||
             std::dynamic_pointer_cast<ngraph::op::CropIE>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Split>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Constant>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Result>(node))) {
                 return false;
        }

        std::string noncomp_prop("non_compute_node");
        std::string result_prop("result_vector");

        auto& rt_info = node->get_rt_info();
        auto res_node = std::dynamic_pointer_cast<ngraph::opset8::Result>(node);
        if (res_node) {
            rt_info[noncomp_prop] = true;
            rt_info[result_prop] = ngraph::ResultVector{res_node};
        }

        if (!rt_info[noncomp_prop].as<bool>())
            return false;

        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto& input_rti = node->get_input_node_shared_ptr(i)->get_rt_info();
            input_rti[noncomp_prop] = true;
            if (input_rti.count(result_prop) && !input_rti[result_prop].as<ngraph::ResultVector>().empty()) {
                for (auto&& res : rt_info[result_prop].as<ngraph::ResultVector>()) {
                    input_rti[result_prop].as<ngraph::ResultVector>().push_back(res);
                }
            } else {
                input_rti[result_prop] = rt_info[result_prop];
            }
        }

        if (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node)) {
            auto result_vec = rt_info[result_prop].as<ngraph::ResultVector>();
            for (auto&& result_node : result_vec) {
                auto copy_out = result_node->get_input_node_shared_ptr(0);
                for (size_t i = 0; i < copy_out->get_input_size(); i++) {
                    auto copy_in = copy_out->get_input_node_shared_ptr(i);
                    if (!std::dynamic_pointer_cast<ngraph::opset8::Constant>(copy_in))
                        InsertCopyLayerBetween(copy_in, copy_out, i);
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(noncompute_op, matcher_name);
    this->register_matcher(m, callback);
}
} // namespace GNAPluginNS
