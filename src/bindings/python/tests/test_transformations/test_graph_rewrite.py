# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime import opset8
from openvino.runtime.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher

from utils.utils import count_ops, get_test_function, PatternReplacement


def test_graph_rewrite():
    model = get_test_function()

    m = Manager()
    # check that register pass returns pass instance
    anchor = m.register_pass(GraphRewrite())
    anchor.add_matcher(PatternReplacement())
    m.run_passes(model)

    assert count_ops(model, "Relu") == [2]


def test_register_new_node():
    class InsertExp(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            param = WrapType("opset8.Parameter")

            def callback(m: Matcher) -> bool:
                # Input->...->Result => Input->Exp->...->Result
                root = m.get_match_value()
                consumers = root.get_target_inputs()

                exp = opset8.exp(root)
                for consumer in consumers:
                    consumer.replace_source_output(exp.output(0))

                # For testing purpose
                self.model_changed = True

                # Use new operation for additional matching
                self.register_new_node(exp)

                # Root node wasn't replaced or changed
                return False

            self.register_matcher(Matcher(param, "InsertExp"), callback)

    class RemoveExp(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            param = WrapType("opset8.Exp")

            def callback(m: Matcher) -> bool:
                root = m.get_match_root()
                root.output(0).replace(root.input_value(0))

                # For testing purpose
                self.model_changed = True

                return True

            self.register_matcher(Matcher(param, "RemoveExp"), callback)

    m = Manager()
    ins = m.register_pass(InsertExp())
    rem = m.register_pass(RemoveExp())
    m.run_passes(get_test_function())

    assert ins.model_changed
    assert rem.model_changed