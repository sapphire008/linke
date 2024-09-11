import pytest
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from kubeflow_components.evaluation.metrics import (
    TopKMetricPreprocessor,
    HitRatioTopK,
    _HitRatioTopKPreprocessor,
    SampleTopKMetricCombiner,
    PopulationTopKMetricCombiner,
)
from kubeflow_components.evaluation.metrics import DEFAULT_PREDICTION_KEY, DEFAULT_LABEL_KEY

from pdb import set_trace


@pytest.mark.skip(reason="temporary")
class TestTopKMetricPreprocessor:
    def setup_method(self, method=None):
        self.preprocessor = TopKMetricPreprocessor(top_k=[1, 4, 5, 10])

    def teardown_method(self, method=None):
        pass

    def test_transform_label_np_ndarray(self):
        label = np.array([[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]])
        transformed, padding = self.preprocessor.transform_label(label)
        assert (transformed == label).all()
        assert padding is None
        label = np.array(
            [["A", "B", "C", "A"], ["B", "D", "A", ""], ["C", "A", "", ""]]
        )
        transformed, padding = self.preprocessor.transform_label(label)
        assert (transformed == label).all()
        assert padding is None

    def test_transform_label_sparse(self):
        label_data = np.array([[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]])
        label = csr_matrix(label_data)
        transformed, padding = self.preprocessor.transform_label(label)
        assert (transformed == label_data).all()
        assert padding == 0
        label.data = np.array(["A", "B", "C", "A", "B", "D", "A", "C", "A"])
        transformed, padding = self.preprocessor.transform_label(label)
        expected_label = np.array(
            [["A", "B", "C", "A"], ["B", "D", "A", ""], ["C", "A", "", ""]]
        )
        assert (transformed == expected_label).all()
        assert padding == ""

    def test_transform_label_list_list(self):
        label = [[1, 2, 3, 1], [2, 4, 1], [3, 1]]
        transformed, padding = self.preprocessor.transform_label(label)
        expected_label = np.array([[1, 2, 3, 1], [2, 4, 1, 0], [3, 1, 0, 0]])
        assert (transformed == expected_label).all()
        assert padding == 0
        label = [["A", "B", "C", "A"], ["B", "D", "A"], ["C", "A"]]
        transformed, padding = self.preprocessor.transform_label(label)
        expected_label = np.array(
            [["A", "B", "C", "A"], ["B", "D", "A", ""], ["C", "A", "", ""]]
        )
        assert (transformed == expected_label).all()
        assert padding == ""

    def test_set_operation(self):
        x = np.array([[5, 0, 0, 0, 0], [2, 1, 4, 0, 0], [3, 2, 0, 0, 0]])
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

@pytest.mark.skip(reason="temporary")
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
        out, n = self.combiner.add_input((accumulator, counter), (state, num))
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


@pytest.mark.skip(reason="temporary")
class TestHitRatioTopK:
    def setup_method(self, method=None):
        self.metric = HitRatioTopK(top_k=[1, 4, 5])

    def test_specs(self):
        assert isinstance(self.metric.combiner, SampleTopKMetricCombiner)
        assert len(self.metric.preprocessors) == 1
        assert isinstance(self.metric.preprocessors[0], _HitRatioTopKPreprocessor)

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
                element[DEFAULT_LABEL_KEY], element[DEFAULT_PREDICTION_KEY]
            ):
                expected += int(len(set(y_true).intersection(y_pred[:k])) > 0)
            assert expected == out_metrics[k]

@pytest.mark.skip(reason="temporary")
class TestPopulationTopKMetricCombiner:
    def setup_method(self, method=None):
        self.combiner = PopulationTopKMetricCombiner(
            metric_key="population_metric", top_k=[1, 4, 5] 
        )

    def test_create_accumulator(self):
        accumulator, counter = self.combiner.create_accumulator()
        assert self.combiner.vocabulary is None
        assert self.combiner.constraint is None
        assert counter == 0
        for k in self.combiner.top_k:
            assert k in accumulator
            assert Counter() == accumulator[k]

    def test_add_input(self):
        accumulator = {
            1: Counter(),
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
        out, n = self.combiner.add_input((accumulator, count), (state, num))
        # check results
        assert n == count + num
        for k in accumulator:
            assert out[k] == accumulator[k] + state[k]

    def test_merge_accumulators(self):
        accumulator1 = ({
            1: Counter(),
            4: Counter({"A": 12, "C": 6}),
            5: Counter({"A": 12, "B": 16, "C": 7}),
        }, 3)
        accumulator2 = ({
            1: Counter({"A": 2, "B": 5,}),
            4: Counter({"A": 3, "B": 7, "C": 3}),
            5: Counter({"B": 9, "C": 4}),
        }, 2)
        accumulator3 = ({
            1: Counter({"A": 1, "C": 2}),
            4: Counter({"A": 3, "B": 7, "C": 3}),
            5: Counter({"A": 3, "B": 9}),
        }, 1)
        accumulators = [accumulator1, accumulator2, accumulator3]
        out, count = self.combiner.merge_accumulators(accumulators)

        total = 0  # total count after merge
        for _, c in accumulators:
            total += c
        assert total == count

        for k in out:
            combined = Counter()  # combined metric
            for acc, _ in accumulators:
                combined = combined + acc[k]
            assert combined == out[k]

    def test_extract_output(self):
        accumulator = {
            1: Counter({"A": 10, "B": 12, "C": 3}),
            4: Counter({"A": 12, "B": 15, "C": 6}),
            5: Counter({"A": 12, "B": 16, "C": 7}),
        }
        num = 10
        output, count = self.combiner.extract_output((accumulator, num))
        assert output == accumulator
        assert count == num
        
        
class TestPopulationTopKMetricCombinerWithVocabulary:
    def setup_method(self, method=None):
        self.combiner = PopulationTopKMetricCombiner(
            metric_key="population_metric", top_k=[1, 4, 5],
            vocabulary=["A", "B", "C", "D", "E"],
            constrain_accumulation=True,
        )
        
    def test_create_accumulator(self):
        accumulator, counter = self.combiner.create_accumulator()
        assert self.combiner.vocabulary is None
        assert self.combiner.constraint is not None
        assert counter == 0
        for k in self.combiner.top_k:
            assert k in accumulator
            # Check all constraints are initialized
            for v in self.combiner.constraint:
                assert v in accumulator[k]
    
    def test_add_input(self):
        accumulator = {
            1: Counter({"A": 0,  "B": 0,  "C": 0, "D": 0, "E": 0}),
            4: Counter({"A": 12, "B": 15, "C": 6, "D": 0, "E": 0}),
            5: Counter({"A": 12, "B": 16, "C": 7, "D": 1, "E": 0}),
        }
        count = 3
        state = {
            1: Counter({"A": 2, "C": 2}),
            4: Counter({"B": 7, "C": 3}),
            5: Counter({"A": 3, "B": 9, "C": 4}),
        }
        num = 2
        out, n = self.combiner.add_input((accumulator, count), (state, num))
        # check results
        assert n == count + num
        for k in accumulator:
            for vv in self.combiner.constraint:
                assert out[k][vv] == accumulator[k][vv] + state[k].get(vv, 0)

    def test_merge_accumulators(self):
        accumulator1 = ({
            1: Counter({"A": 0,  "B": 0,  "C": 0, "D": 0, "E": 0}),
            4: Counter({"A": 12, "B": 15, "C": 6, "D": 0, "E": 0}),
            5: Counter({"A": 12, "B": 16, "C": 7, "D": 1, "E": 0}),
        }, 3)
        accumulator2 = ({
            1: Counter({"A": 2, "B": 5, "C": 0, "D": 0, "E": 0}),
            4: Counter({"A": 3, "B": 7, "C": 3, "D": 0, "E": 0}),
            5: Counter({"A": 0, "B": 9, "C": 4, "D": 0, "E": 0}),
        }, 2)
        accumulator3 = ({
            1: Counter({"A": 1, "B": 0, "C": 2, "D": 0, "E": 0}),
            4: Counter({"A": 3, "B": 7, "C": 3, "D": 0, "E": 0}),
            5: Counter({"A": 3, "B": 9, "C": 0, "D": 0, "E": 0}),
        }, 1)
        accumulators = [accumulator1, accumulator2, accumulator3]
        out, count = self.combiner.merge_accumulators(accumulators)

        total = 0  # total count after merge
        for _, c in accumulators:
            total += c
        assert total == count

        combined = {k: {} for k in out}  # combined metric
        for k in out:
            for acc, _ in accumulators:
                for vv in self.combiner.constraint:
                    combined[k][vv] = combined[k].get(vv, 0) + acc[k][vv]
        for k in out:
            for vv in self.combiner.constraint:
                assert combined[k][vv] == out[k][vv]
        