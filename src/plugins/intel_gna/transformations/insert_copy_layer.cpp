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

namespace GNAPluginNS {

NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcat, "HandleMultiConnectedLayerToConcat", 0);
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeConcatLayer, "InsertCopyBeforeConcatLayer", 0);
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeMemoryLayer, "InsertCopyBeforeMemoryLayer", 0);
NGRAPH_RTTI_DEFINITION(HandleLayerConnectedToMultipleConcatsOrMemories, "HandleLayerConnectedToMultipleConcatsOrMemories", 0);

namespace {
    void InsertCopyLayerBetween(std::shared_ptr<ngraph::Node> input_op,
                                std::shared_ptr<ngraph::Node> output_op,
                                const size_t& index) {
        NGRAPH_CHECK(input_op);
        NGRAPH_CHECK(output_op);

        auto copy_op = std::make_shared<GNAPluginNS::Copy>(input_op->output(output_op->input(index).get_source_output().get_index()), false);
        copy_op->set_friendly_name(input_op->get_friendly_name() + "/copy_layer/" + output_op->get_friendly_name() + "." + std::to_string(index));
        ngraph::copy_runtime_info(input_op, copy_op);

        output_op->input(index).replace_source_output(copy_op);
    }

    bool IsNonFunctionalGNANode(std::shared_ptr<ngraph::Node> node) {
        return std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Squeeze>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(node);
    }

    std::shared_ptr<ngraph::opset8::StridedSlice> ConstructStridedSlice() {
        auto data = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
        auto m_begin = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{2});
        auto m_end = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{2});
        auto m_stride = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{2});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        return std::make_shared<ngraph::opset8::StridedSlice>(data, m_begin, m_end, m_stride, begin_mask, end_mask);
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
            auto input_op = concat->input(i).get_source_output().get_node_shared_ptr();

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

    // auto slice_in = ConstructStridedSlice();
    // auto crop_in = ngraph::pattern::wrap_type<ngraph::op::CropIE>();
    auto split_in = ngraph::pattern::wrap_type<ngraph::opset8::Split>();
    // auto vsplit_in = ngraph::pattern::wrap_type<ngraph::opset8::VariadicSplit>();
    // auto concat_in = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    // auto input_ops = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{slice_in, crop_in, split_in, vsplit_in, concat_in});

    auto read_op = std::make_shared<ngraph::opset3::ReadValue>({split_in});
    auto assign_op = std::make_shared<ngraph::opset3::Assign>({split_in});
    auto memory_ops = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{read_op, assign_op});

    // crop/split/concat -> memory
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        auto memory_iter = pattern_map.find(assign_op);
        if (memory_iter == pattern_map.end() &&
           (memory_iter = pattern_map.find(read_op)) == pattern_map.end())
            return false;

        auto memory_node = memory_iter->second.get_node_shared_ptr();
        auto memory_input = memory_node->input(0).get_source_output().get_node_shared_ptr();
        InsertCopyLayerBetween(memory_input, memory_node, 0);

        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(memory_ops, matcher_name);
    this->register_matcher(m, callback);
}

InsertCopyBeforeConcatLayer::InsertCopyBeforeConcatLayer() {
    MATCHER_SCOPE(InsertCopyBeforeConcatLayer);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();

    auto match_ops = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{concat_op});
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());

        if (!std::dynamic_pointer_cast<ngraph::opset8::Concat>(node))
                return false;

        // crop/split -> concat
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto curr_input = node->input(i).get_source_output().get_node_shared_ptr();

            bool is_concat_case = std::dynamic_pointer_cast<ngraph::opset8::Split>(curr_input) ||
                std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(curr_input) ||
                std::dynamic_pointer_cast<ngraph::opset8::StridedSlice>(curr_input) ||
                std::dynamic_pointer_cast<ngraph::op::CropIE>(curr_input);

            if (!is_concat_case) {
                continue;
            }

            InsertCopyLayerBetween(curr_input, node, i);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(match_ops, matcher_name);
    this->register_matcher(m, callback);
}

// TODO HANDLE TRIVIAL PERMUTE
bool HandleLayerConnectedToMultipleConcatsOrMemories::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleLayerConnectedToMultipleConcatsOrMemories);
    bool is_graph_modified = false;
    for (auto& node : f->get_ordered_ops()) {
        for (auto& output : node->outputs()) {
            auto inputTo = output.get_target_inputs();
            if (inputTo.size() < 2) continue;
            std::vector<std::pair<std::shared_ptr<ngraph::Node>, size_t>> concat_nodes, memory_nodes;
            for (auto& child : inputTo) {
                auto current_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto copy_output_node = current_node;
                auto previous_node = node;
                auto current_index = child.get_index();

                while ((IsNonFunctionalGNANode(current_node)) && current_node->get_output_size() == 1) {
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
} // namespace GNAPluginNS
