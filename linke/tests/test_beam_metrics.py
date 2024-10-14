import os
import pytest
import copy
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from pdb import set_trace

from linke.evaluation.beam.metrics import (
    TopKMetricPreprocessor,
    HitRatioTopK,
    _HitRatioTopKPreprocessor,
    NDCGTopK,
    _NDCGTopKPreprocessor,
    SampleTopKMetricCombiner,
    PopulationTopKMetricCombiner,
    PopulationTopKMetricPreprocessor,
    _CoverageTopKCombiner,
    CoverageTopK,
    UniqueCountTopK,
    _UniqueCountTopKPreprocessor,
    MiscalibrationTopK,
    _MiscalibrationTopKPreprocessor,
)
from linke.evaluation.beam.metrics import (
    DEFAULT_PREDICTION_KEY,
    DEFAULT_LABEL_KEY,
    DEFAULT_FEATURE_KEY,
)


class TestTopKMetricPreprocessor:
    def setup_method(self, method=None):
        self.preprocessor = TopKMetricPreprocessor(top_k=[1, 4, 5, 10])

    def teardown_method(self, method=None):
        pass

    def test_sparse_to_dense_np_ndarray(self):
        label = np.array([[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]])
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        assert (transformed == label).all()
        assert padding is None
        label = np.array(
            [
                ["A", "B", "C", "A"],
                ["B", "D", "A", ""],
                ["C", "A", "", ""],
            ]
        )
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        assert (transformed == label).all()
        assert padding is None

    def test_sparse_to_dense_sparse(self):
        label_data = np.array(
            [[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]]
        )
        label = csr_matrix(label_data)
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        assert (transformed == label_data).all()
        assert padding == 0
        label.data = np.array(
            ["A", "B", "C", "A", "B", "D", "A", "C", "A"]
        )
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        expected_label = np.array(
            [
                ["A", "B", "C", "A"],
                ["B", "D", "A", ""],
                ["C", "A", "", ""],
            ]
        )
        assert (transformed == expected_label).all()
        assert padding == ""

    def test_sparse_to_dense_list_list(self):
        label = [[1, 2, 3, 1], [2, 4, 1], [3, 1]]
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        expected_label = np.array(
            [[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]]
        )
        assert (transformed == expected_label).all()
        assert padding == 0
        label = [["A", "B", "C", "A"], ["B", "D", "A"], ["C", "A"]]
        transformed, padding = self.preprocessor.sparse_to_dense(label)
        expected_label = np.array(
            [
                ["A", "B", "C", "A"],
                ["B", "D", "A", ""],
                ["C", "A", "", ""],
            ]
        )
        assert (transformed == expected_label).all()
        assert padding == ""

    def test_set_operation(self):
        x = np.array(
            [[5, 0, 0, 0, 0], [2, 1, 4, 0, 0], [3, 2, 0, 0, 0]]
        )
        y = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 1, 2, 4]])
        # intersection
        out = self.preprocessor.set_operation(
            x, y, pad=0, operation="intersection", returns="count"
        )
        assert (out == np.array([0, 3, 2])).all()
        # union
        out = self.preprocessor.set_operation(
            x, y, pad=0, operation="union", returns="count"
        )
        assert (out == np.array([5, 4, 4])).all()
        # difference
        out = self.preprocessor.set_operation(
            y, x, pad=0, operation="difference", returns="count"
        )
        assert (out == np.array([4, 1, 2])).all()
        # intersection, return matrix
        out = self.preprocessor.set_operation(
            x, y, pad=0, operation="intersection", returns="matrix"
        )
        assert isinstance(out, csr_matrix)
        assert (out.data == np.array([4, 1, 2, 2, 3])).all()
        assert (out.tocoo().row == np.array([1, 1, 1, 2, 2])).all()
        assert (out.tocoo().col == np.array([0, 1, 2, 0, 1])).all()


class TestSampleTopKMetricCombiner:
    """Test beam.CombineFn"""

    def setup_method(self, method):
        self.combiner = SampleTopKMetricCombiner(
            metric_key="sample_metric", top_k=[1, 4, 5]
        )

    def test_create_accumulator(self):
        accumulator, counter = self.combiner.create_accumulator()
        assert counter == 0
        for k in self.combiner.top_k:
            assert k in accumulator
        assert np.allclose(list(accumulator.values()), 0.0)

    def test_add_input(self):
        accumulator, counter = {1: 0.8, 4: 1.4, 5: 2.1}, 3
        state, num = {1: 2.1, 4: 3.6, 5: 7.6}, 5
        out, n = self.combiner.add_input(
            (accumulator, counter), (state, num)
        )
        # check results
        assert n == counter + num
        for k in accumulator:
            assert np.allclose(out[k], accumulator[k] + state[k])

    def test_merge_accumulators(self):
        accumulator1 = ({1: 0.8, 4: 1.4, 5: 2.1}, 3)
        accumulator2 = ({1: 2.1, 4: 3.6, 5: 7.6}, 5)
        accumulator3 = ({1: 0.3, 4: 0.6, 5: 1.4}, 2)
        accumulators = [accumulator1, accumulator2, accumulator3]
        out, count = self.combiner.merge_accumulators(accumulators)

        total = 0  # total count after merge
        for _, c in accumulators:
            total += c
        assert total == count

        for k in out:
            combined = 0  # combined metric
            for acc, _ in accumulators:
                combined += acc[k]
            assert np.allclose(combined, out[k])

    def test_extract_output(self):
        accumulator, count = {1: 5.7, 4: 10.2, 5: 11.8}, 9
        out: dict = self.combiner.extract_output((accumulator, count))
        for k, v in out.items():
            np.allclose(v, accumulator[k] / count)


class TestHitRatioTopK:
    def setup_method(self, method=None):
        self.metric = HitRatioTopK(top_k=[1, 4, 5])

    def test_specs(self):
        assert isinstance(
            self.metric.combiner, SampleTopKMetricCombiner
        )
        assert len(self.metric.preprocessors) == 1
        assert isinstance(
            self.metric.preprocessors[0], _HitRatioTopKPreprocessor
        )

    def test_processor(self):
        processor = self.metric.preprocessors[0]
        element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "G", "D", "E"],
                    ["D", "B", "C", "A", "E", "F"],
                    ["F", "A", "E", "B", "C", "H"],
                ]
            ),
            DEFAULT_LABEL_KEY: [["A", "F"], ["G", "H", "C"], ["D"]],
        }
        out_metrics, out_num = next(iter(processor.process(element)))
        # Check num elements expected
        assert out_num == len(element[DEFAULT_LABEL_KEY])
        for k in self.metric.combiner.top_k:
            expected = 0
            # iterative implementation of hit ratio
            for y_true, y_pred in zip(
                element[DEFAULT_LABEL_KEY],
                element[DEFAULT_PREDICTION_KEY],
            ):
                expected += int(
                    len(set(y_true).intersection(y_pred[:k])) > 0
                )
            assert expected == out_metrics[k]


class TestNDCGTopK:
    def setup_method(self, method=None):
        self.metric = NDCGTopK(top_k=[1, 4, 5], weight_key="weight")

    def test_specs(self):
        assert isinstance(
            self.metric.combiner, SampleTopKMetricCombiner
        )
        assert len(self.metric.preprocessors) == 1
        assert isinstance(
            self.metric.preprocessors[0], _NDCGTopKPreprocessor
        )

    def compare_results(self, expected_rel, out_metrics):

        for k in self.metric.combiner.top_k:
            expected = 0
            # iterative implementation of hit ratio
            for rel in expected_rel:
                dcg = (2 ** rel[:k] - 1) / np.log2(np.arange(k) + 2)
                dcg = np.sum(dcg)
                ideal = np.sort(rel[:k])[::-1]
                idcg = (2 ** ideal[:k] - 1) / np.log2(np.arange(k) + 2)
                idcg = np.sum(idcg)
                expected += dcg / (idcg if idcg > 0.0 else 1.0)
            assert np.allclose(expected, out_metrics[k])

    def test_processor_binary(self):
        processor = self.metric.preprocessors[0]
        # Binary case
        element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "G", "F"],
                    ["B", "A", "D", "H", "F"],
                    ["C", "B", "A", "G", "H"],
                ]
            ),
            DEFAULT_LABEL_KEY: [
                ["A", "C"],
                ["B"],
                ["D", "E", "F"],
            ],
        }
        expected_rel = np.array(
            [
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        out_metrics, out_num = next(iter(processor.process(element)))

        # Check num elements expected
        assert out_num == len(element[DEFAULT_LABEL_KEY])
        self.compare_results(expected_rel, out_metrics)

    def test_processor_weighted(self):
        processor = self.metric.preprocessors[0]
        # Binary case
        element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "G", "F"],
                    ["B", "A", "D", "H", "F"],
                    ["C", "B", "A", "G", "H"],
                ]
            ),
            DEFAULT_LABEL_KEY: [
                ["A", "C"],
                ["B"],
                ["D", "E", "F"],
            ],
            "weight": [[0.2, 0.4], [0.9], [0.1, 0.3, 0.5]],
        }
        expected_rel = np.array(
            [
                [0.2, 0, 0.4, 0, 0],
                [0.9, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        out_metrics, out_num = next(iter(processor.process(element)))

        # Check num elements expected
        assert out_num == len(element[DEFAULT_LABEL_KEY])
        self.compare_results(expected_rel, out_metrics)


class TestPopulationTopKMetricPreprocessor:
    def setup_method(self, method=None):
        self.preprocessor = PopulationTopKMetricPreprocessor(
            top_k=[1, 4, 5]
        )
        self.preprocessor_vocab = PopulationTopKMetricPreprocessor(
            top_k=[1, 4, 5],
            vocabulary_fields=["label", "prediction", "history"],
        )

    def test_process(self):
        element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "G", "D", "E"],
                    ["D", "B", "C", "A", "E", "F"],
                    ["F", "A", "E", "B", "C", "H"],
                ]
            ),
            DEFAULT_LABEL_KEY: [["A", "F"], ["G", "H", "C"], ["D"]],
            "history": [["A"], [], ["B", "C"]],
        }
        result, _ = next(iter(self.preprocessor.process(element)))
        assert "_vocab" not in result

        result, _ = next(iter(self.preprocessor_vocab.process(element)))
        assert "_vocab" in result


class TestPopulationTopKMetricCombiner:
    def setup_method(self, method=None):
        self.combiner_default = PopulationTopKMetricCombiner(
            metric_key="population_metric_combiner",
            top_k=[1, 4, 5],
        )
        self.combiner_no_vocab = PopulationTopKMetricCombiner(
            metric_key="population_metric_combiner",
            top_k=[1, 4, 5],
            retain_zeros=True,
        )
        self.combiner_with_vocab = PopulationTopKMetricCombiner(
            metric_key="population_metric_combiner_with_vocabulary",
            top_k=[1, 4, 5],
            vocabulary=["A", "B", "C", "D", "E"],
        )

    def test_create_accumulator(self):
        assert self.combiner_default.vocabulary is None
        assert self.combiner_no_vocab.vocabulary is None
        assert self.combiner_with_vocab.vocabulary is not None
        accumulator, counter = (
            self.combiner_no_vocab.create_accumulator()
        )
        assert counter == 0
        for k in (
            self.combiner_no_vocab.top_k
            + self.combiner_no_vocab.other_fields
        ):
            assert k in accumulator
            assert Counter() == accumulator[k]

    def test_add_input(self):
        accumulator = {
            1: Counter({"B": 0}),
            4: Counter({"A": 12, "B": 15, "C": 6}),
            5: Counter({"A": 12, "B": 16, "C": 7}),
        }
        count = 3
        state = {
            1: Counter({"A": 2, "C": 2}),
            4: Counter({"B": 7, "C": 3}),
            5: Counter({"A": 3, "B": 9, "C": 4}),
        }
        num = 2
        out1, _ = self.combiner_default.add_input(
            (accumulator, count), (state, num)
        )
        assert "B" not in out1[1]

        out2, n = self.combiner_with_vocab.add_input(
            (accumulator, count), (state, num)
        )
        # check results
        assert n == count + num

        for k in accumulator:
            # print(out1[k], accumulator[k], state[k])
            assert out2[k] == accumulator[k] + state[k]

        out3, _ = self.combiner_no_vocab.add_input(
            (accumulator, count), (state, num)
        )
        # all keys includig zero keys are present
        assert all([key in out3[1] for key in ["A", "B", "C"]])

    def test_merge_accumulators(self):
        accumulator1 = (
            {
                1: Counter({"D": 0}),
                4: Counter({"A": 12, "C": 6}),
                5: Counter({"A": 12, "B": 16, "C": 7}),
            },
            3,
        )
        accumulator2 = (
            {
                1: Counter({"A": 2, "B": 5}),
                4: Counter({"A": 3, "B": 7, "C": 3}),
                5: Counter({"B": 9, "C": 4}),
            },
            2,
        )
        accumulator3 = (
            {
                1: Counter({"A": 1, "C": 2}),
                4: Counter({"A": 3, "B": 7, "C": 3}),
                5: Counter({"A": 3, "B": 9}),
            },
            1,
        )
        accumulators = [accumulator1, accumulator2, accumulator3]
        out, count = self.combiner_default.merge_accumulators(
            accumulators
        )
        assert "D" not in out[1]

        total = 0  # total count after merge
        for _, c in accumulators:
            total += c
        assert total == count

        # When no vocab is present
        out2, _ = self.combiner_no_vocab.merge_accumulators(
            accumulators
        )
        assert "D" in out2[1] and out2[1]["D"] == 0

        out3, _ = self.combiner_with_vocab.merge_accumulators(
            accumulators
        )

        for k in out3:
            combined = Counter()  # combined metric
            for acc, _ in accumulators:
                combined = combined + acc[k]
            assert combined == out3[k]
        assert "D" not in out3[1]

    def test_extract_output(self):
        accumulator = {
            1: Counter({"A": 10, "B": 12, "C": 3}),
            4: Counter({"A": 12, "B": 15, "C": 6}),
            5: Counter({"A": 12, "B": 16, "C": 7}),
        }
        num = 10
        output, count = self.combiner_with_vocab.extract_output(
            (accumulator, num)
        )
        assert output == accumulator
        assert count == num


# @pytest.mark.skip(reason="")
class TestCoverageTopK:
    def setup_method(self, method=None):
        self.metric = CoverageTopK(
            top_k=[1, 4, 5],
            include_labels=True,
            vocabulary=None,
            # estimate vocabulary using "prediction"
        )
        self.metric_vocabulary = CoverageTopK(
            top_k=[1, 4, 5],
            include_labels=True,
            vocabulary=["A", "B", "C", "D", "E", "F"],
        )
        self.element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "D", "E"],
                    ["D", "A", "B", "C", "E"],
                    ["A", "C", "B", "D", "E"],
                ]
            ),
            DEFAULT_LABEL_KEY: [
                ["A", "C"],
                ["B"],
                ["D", "E", "F"],
            ],
        }

    def test_specs(self):
        assert isinstance(self.metric.combiner, _CoverageTopKCombiner)
        assert len(self.metric.preprocessors) == 1
        assert isinstance(
            self.metric.preprocessors[0],
            PopulationTopKMetricPreprocessor,
        )
        # By default if no vocab is specified, "prediction"
        # is used to estimate vocabulary
        assert self.metric.combiner.vocabulary is None
        vocab_estimate = self.metric.preprocessors[0].vocabulary_fields
        assert len(vocab_estimate) == 1
        assert vocab_estimate[0] == "prediction"

        assert self.metric_vocabulary.combiner.vocabulary is not None

    def test_metric(self):
        processor = self.metric.preprocessors[0]
        combiner = self.metric.combiner

        # Check preprocessor
        out_metrics, out_num = next(
            iter(processor.process(self.element))
        )
        # Check num elements expected
        assert out_num == len(self.element[DEFAULT_LABEL_KEY])
        for k in out_metrics:
            if k == "_vocab":
                assert len(out_metrics[k]) == 5
                continue
            elif k == "label":
                expected = Counter(
                    sum(self.element[DEFAULT_LABEL_KEY], [])
                )
            else:
                expected = Counter(
                    self.element[DEFAULT_PREDICTION_KEY][:, :k].ravel()
                )

            assert expected == out_metrics[k]

        # Check combiner
        output = combiner.extract_output((out_metrics, out_num))
        # check: if we forget to take the intersection between
        # label and accumulated vocabulary, the "label" value
        # would end up 1.2, which may not make sense.
        expected = {1: 0.4, 4: 0.8, 5: 1.0, "label": 1.0}
        assert output == expected

    def test_metric_vocabulary(self):
        processor = self.metric_vocabulary.preprocessors[0]
        combiner = self.metric_vocabulary.combiner

        # Check preprocessor
        out_metrics, out_num = next(
            iter(processor.process(self.element))
        )
        # Check num elements expected
        assert out_num == len(self.element[DEFAULT_LABEL_KEY])
        assert "_vocab" not in out_metrics
        for k in out_metrics:
            if k == "label":
                expected = Counter(
                    sum(self.element[DEFAULT_LABEL_KEY], [])
                )
            else:
                expected = Counter(
                    self.element[DEFAULT_PREDICTION_KEY][:, :k].ravel()
                )

            assert expected == out_metrics[k]

        # Check combiner
        output = combiner.extract_output((out_metrics, out_num))
        expected = {1: 2 / 6, 4: 4 / 6, 5: 5 / 6, "label": 1.0}
        assert all(
            [np.allclose(output[k], expected[k]) for k in output]
        )


class TestUniqueCountTopK:
    pass


class TestMiscalibrationTopK:
    def setup_method(self, method=None):
        tag_maps={
                "A": ["tag1", "tag2", "tag3"],
                "B": ["tag3", "tag5"],
                "C": ["tag4", "tag6", "tag1", "tag2"],
                "D": ["tag2"],
                "E": ["tag4", "tag1", "tag2"],
                "F": ["tag3", "tag1"],
                "G": ["tag5"],
                "H": ["tag3", "tag4"],
                "I": ["tag1", "tag2", "tag3", "tag4", "tag7"],
            }
        self.metric = MiscalibrationTopK(
            top_k=[1, 4, 5],
            tag_maps=tag_maps,
            history_feature="view_history",
            distance_metric="hellinger",
        )
        
        self.element = {
            DEFAULT_PREDICTION_KEY: np.array(
                [
                    ["A", "B", "C", "G", "D"],
                    ["D", "B", "I", "A", "E"],
                    ["F", "E", "G", "C", "H"],
                ]
            ),
            DEFAULT_FEATURE_KEY: {
                "view_history": [
                    ["H", "I"],
                    ["C", "G", "H"],
                    ["A", "B"],
                ]
            }
        }

    def test_specs(self):
        assert isinstance(
            self.metric.combiner, SampleTopKMetricCombiner
        )
        assert len(self.metric.preprocessors) == 1
        assert isinstance(
            self.metric.preprocessors[0],
            _MiscalibrationTopKPreprocessor,
        )

    def test_loading_tag_maps(self):
        metric = MiscalibrationTopK(
            top_k=[1, 4, 5],
            tag_maps="linke/tests/data/tag_map.txt",
            history_feature="view_history",
        )
        metric.preprocessors[0].setup()
        tag_maps = metric.preprocessors[0].tag_maps
        assert isinstance(tag_maps, dict)
        assert len(tag_maps.keys()) == 9

    def test_processor_hellinger(self):
        processor: _MiscalibrationTopKPreprocessor = self.metric.preprocessors[0]
        out_metrics, out_num = next(iter(processor.process(self.element)))
        assert out_num == len(self.element[DEFAULT_PREDICTION_KEY])
        assert all([k in out_metrics for k in self.metric.combiner.top_k])
        # Using for-loop, one at a time
        tags_map = self.metric.preprocessors[0].tag_maps
        
        def _hellinger_distance(history, prediction):
            tag_history = Counter(sum([tags_map[h] for h in history], []))
            tag_pred = Counter(sum([tags_map[p] for p in prediction], []))
            all_tags = tag_history+tag_pred
            p, q = [], []
            for key in all_tags:
                p.append(tag_history.get(key, 0))
                q.append(tag_pred.get(key, 0))
            p = np.asarray(p)
            q = np.asarray(q)
            p = p / np.sum(p)
            q = q / np.sum(q)
            dist = np.sum((np.sqrt(p) - np.sqrt(q))**2)
            return np.sqrt(dist) / np.sqrt(2)

        for k in self.metric.combiner.top_k:
            res = 0
            for ii in range(3):
                history = self.element[DEFAULT_FEATURE_KEY]["view_history"][ii]
                prediction = self.element[DEFAULT_PREDICTION_KEY][ii][:k]
                res += _hellinger_distance(history, prediction)
            assert np.allclose(out_metrics[k], res)
    
    def test_preprocessor_kl(self):
        processor: _MiscalibrationTopKPreprocessor = self.metric.preprocessors[0]
        processor.distance_metric = "kl-divergence"
        out_metrics, out_num = next(iter(processor.process(self.element)))
        tags_map = self.metric.preprocessors[0].tag_maps
        eps = _MiscalibrationTopKPreprocessor.eps
        
        def _kl_divergence(history, prediction):
            tag_history = Counter(sum([tags_map[h] for h in history], []))
            tag_pred = Counter(sum([tags_map[p] for p in prediction], []))
            all_tags = tag_history+tag_pred
            p, q = [], []
            for key in all_tags:
                p.append(tag_history.get(key, 0))
                q.append(tag_pred.get(key, 0))
            p = np.asarray(p)
            q = np.asarray(q)
            p = p / np.sum(p)
            q = q / np.sum(q)
            dist = p * (np.log(p + eps) - np.log(q + eps))
            return dist.sum()
        
        for k in self.metric.combiner.top_k:
            res = 0
            for ii in range(3):
                history = self.element[DEFAULT_FEATURE_KEY]["view_history"][ii]
                prediction = self.element[DEFAULT_PREDICTION_KEY][ii][:k]
                res += _kl_divergence(history, prediction)
            assert np.allclose(out_metrics[k], res, atol=1e-3)
        
