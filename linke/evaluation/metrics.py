"""Implement recommendation system top-k metrics"""

from typing import (
    List,
    Dict,
    Any,
    Optional,
    Literal,
    Generator,
    Union,
    Tuple,
    Iterable,
)
import copy
from itertools import zip_longest
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, sparray
import apache_beam as beam
from apache_beam.transforms.stats import ApproximateUnique, ApproximateUniqueCombineFn

from pdb import set_trace


# %% Metrics Base Classes
DEFAULT_PREDICTION_KEY = "prediction"
DEFAULT_LABEL_KEY = "label"
DEFAULT_FEATURE_KEY = "feature"


class BaseMetric:
    """Similar to tfma.metrics.Metric"""

    def __init__(
        self,
        name: str,
        preprocessors: Iterable[beam.DoFn],
        combiner: Union[beam.CombineFn, Dict[str, beam.CombineFn]],
    ):
        self.name = name
        self.preprocessors = preprocessors
        # Either a single CombineFn or a dictionary of combine funcs
        # where the key is corresponding to the key in the output of the
        # final preprocessor. If this format is used, then the output
        # of the final preprocessor must be a dictionary.
        self.combiner = combiner


class TopKMetricPreprocessor(beam.DoFn):
    """Helper class to define preprocessor beam.DoFn"""

    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
        group_keys: List[List[str]] = None,
    ):
        """
        Base class for TopK Metrics

        Args:
            top_k (Union[int, List[int]]): top k or
                list of top k to limit the top predictions.
            feature_key (str, optional): key to the features
                in the extracted data dictionary.
                Defaults to DEFAULT_FEATURE_KEY.
            prediction_key (str, optional): key to the predictions
                in the extracted data dictionary.
                Defaults to DEFAULT_PREDICTION_KEY.
            label_key (str, optional): key to the labels
                in the extracted data dictionary.
                Defaults to DEFAULT_LABEL_KEY.
            weight_key (str, optional): key to the weights
                in the extracted data dictionary. Defaults to None.
            group_keys (List[List[str]], optional): list of list of keys
                to group the aggregation by. Defaults to None.
        """
        super().__init__()
        self.top_k = [top_k] if isinstance(top_k, int) else top_k
        self.feature_key = feature_key or DEFAULT_FEATURE_KEY
        self.prediction_key = prediction_key or DEFAULT_PREDICTION_KEY
        self.label_key = label_key or DEFAULT_LABEL_KEY
        self.weight_key = weight_key
        self.group_keys = group_keys

    @staticmethod
    def sparse_to_dense(
        label: Union[np.ndarray, sparray, List[List]],
        pad: Optional[Union[int, float, str]] = None,
    ):
        """Convert a sparse tensor to a padded dense tensor."""
        # Handle np.ndarray type
        if isinstance(label, np.ndarray):
            return label, None

        # Handle list type
        if isinstance(label, list):
            if isinstance(label[0][0], int):
                fillvalue = 0 if pad is None else pad
                if pad is None:
                    fillvalue = min([min(arr) for arr in label]) - 1
                else:
                    fillvalue = pad
            elif isinstance(label[0][0], float):
                fillvalue = 0.0 if pad is None else pad
            elif isinstance(label[0][0], str):
                fillvalue = "" if pad is None else pad
            else:
                raise (
                    TypeError(
                        f"Unable to handle sparse label with dtype {type(label[0][0])}"
                    )
                )
            label = np.array(list(zip_longest(*label, fillvalue=fillvalue))).T
            return label, fillvalue

        # Create dense tensor label
        # For integer matrix, use min_val - 1 as padding
        label: sparray = label
        if np.issubdtype(label.data.dtype, np.integer):
            if pad is None:
                fillvalue = label.data.min() - 1
            else:
                fillvalue = pad
            label: coo_matrix = label.tocoo()
            res = np.full(label.shape, fillvalue, dtype=label.data.dtype)
            np.put(res, label.row * label.shape[1] + label.col, label.data)
            return res, fillvalue

        # For float matrix, use 0 as padding
        if np.issubdtype(label.data.dtype, np.floating):
            return label.todense(), 0.0

        # For string matrix, use "" by default
        if np.issubdtype(label.data.dtype, np.object_) or np.issubdtype(
            label.data.dtype, np.str_
        ):
            label: coo_matrix = label.tocoo()
            res = np.full(label.shape, "", dtype=label.data.dtype)
            np.put(res, label.row * label.shape[1] + label.col, label.data)
            return res, ""

        # Throw exception for all other label dtypes
        raise (
            TypeError(f"Unable to handle sparse label with dtype {label.data.dtype}")
        )

    @staticmethod
    def set_operation(
        x: np.ndarray,
        y: np.ndarray,
        pad: Union[int, str] = None,
        operation: Literal["intersection", "union", "difference"] = "intersection",
        returns: Literal["count", "matrix"] = "count",
    ):
        """
        Batch-wise set operations.

        Inputs:
            * x, y: dense 2D np.ndarrays with / without padding values.
            * pad: padding values. Default to None which is no padding.
            * operation: set operation to perform. Valid values are
                ["intersection" (default), "union", "difference"].
            * returns: type of returned outputs, either "count" (default) which
                counts the cardinality of the resulting set operation for each
                row or "matrix" which returns the sparse matrix output from
                the set operations, all values left aligned.
        """
        # Input shapes
        n_d, m_x = x.shape
        n_d, m_y = y.shape

        # Use np.unique to create convert from data -> indices
        # This can appropriately handle all data types, including strings
        unique, indices = np.unique(np.hstack((x, y)), return_inverse=True)
        n_unique = len(unique)

        # From flattened index -> original shape
        indices = indices.reshape(n_d, -1)
        indices_x = indices[:, :m_x]
        indices_y = indices[:, m_x:]

        # which index from unique is the padding of the ragged representation
        pad_index = np.where(unique == pad)[0]
        if len(pad_index) > 0:  # found the padding
            pad_index = pad_index[0]
        else:
            pad_index = -1

        # Use csr format to create to create binary indicator matrices
        # e.g. index = [1, 3], n_unique = 5 -> [0, 1, 0, 1, 0]
        def _create_csr_from_dense_idx(idx, m):
            # create csr
            indptr = np.repeat([m], n_d).cumsum()
            indptr = np.concatenate([[0], indptr])
            indices = idx.ravel()  # flatten
            data = np.ones_like(indices, dtype=int)
            data[indices == pad_index] = 0  # filter out pad index
            sparse_matrix = csr_matrix(
                (data, indices, indptr),
                shape=(n_d, n_unique),
                dtype=int,
            )
            # eliminate padding if any
            sparse_matrix.eliminate_zeros()
            return sparse_matrix

        x_hat = _create_csr_from_dense_idx(indices_x, m_x)
        y_hat = _create_csr_from_dense_idx(indices_y, m_y)

        # set operations in binary
        if operation == "intersection":
            res = x_hat.multiply(y_hat)
        elif operation == "union":
            res = x_hat + y_hat
            res.data = np.minimum(res.data, 1)
        elif operation == "difference":
            res = x_hat - y_hat
            res.data = np.maximum(res.data, 0)
        else:
            raise (ValueError(f"Unrecognized operation {operation}"))

        if returns == "count":  # return cardinality of set
            return res.sum(axis=1).A.ravel()
        else:  # return the actual sparse matrix
            # keep only entries of 1s
            res.eliminate_zeros()
            # replace the indicator back to actual intersected values
            res.data = np.take(unique, res.indices)
            # left align
            l = np.diff(res.indptr)  # array lengths
            cum_indices = np.arange(res.indptr[-1])
            offsets = np.repeat(res.indptr[:-1], l)
            res.indices = cum_indices - offsets

            return res

    @staticmethod
    def flatten_array(
        label: Union[np.ndarray, sparray, List[List]],
        pad: Optional[Union[int, float, str]] = None,
    ) -> np.ndarray:
        if isinstance(label, np.ndarray):
            flattened = label.ravel()
            if pad is None:
                return flattened
            flattened = flattened[flattened != pad]
            return flattened

        if isinstance(label, list):
            return np.array([x for xs in label for x in xs])

        label: sparray = label
        return label.tocsr().data

    def with_group_keys(self, group_keys: Optional[List[str]] = None):
        if not group_keys:
            return self
        instance = copy.deepcopy(self)
        instance.group_keys = group_keys
        return instance

    def accumulate_metric(self, element):
        """To be overwritten."""
        return element

    def process(self, element: Dict):
        for output in self.accumulate_metric(element):
            if self.group_keys:
                # Assuming, when group_keys i not None,
                # the pipeline from inference only ouputs
                # one example at a time
                feature = element[self.feature_key]
                groupbys = []
                # Attempt to attach multiple grouping keys
                for keys in self.group_keys:  # each slice
                    groupby = [feature[key][0] for key in keys]
                    groupbys.append(tuple(groupby))
                # Add group keys
                output = (tuple(groupbys), output)
            yield output


# %% Sample-wise Metrics
class SampleTopKMetricCombiner(beam.CombineFn):
    def __init__(self, metric_key: str, top_k: Union[int, List[int]]):
        super(SampleTopKMetricCombiner, self).__init__()
        self.metric_key = metric_key
        self.top_k = [top_k] if isinstance(top_k, int) else top_k

    def create_accumulator(self) -> Tuple[Dict[int, float], int]:
        # top-k accumulator, count
        return {k: 0.0 for k in self.top_k}, 0

    def add_input(
        self,
        accumulator: Tuple[Dict[int, float], int],
        state: Tuple[Dict[int, float], int],
    ) -> Tuple[Dict[int, float], int]:
        metrics = {k: accumulator[0].get(k, 0.0) + state[0][k] for k in state[0]}
        n = accumulator[1] + state[1]
        return metrics, n

    def merge_accumulators(
        self, accumulators: Iterable[Tuple[Dict[int, float], int]]
    ) -> Tuple[Dict[int, float], int]:
        accumulators = iter(accumulators)
        result = next(accumulators)  # get the first item
        for accumulator in accumulators:
            metric = {k: accumulator[0].get(k, 0.0) + result[0][k] for k in result[0]}
            n = accumulator[1] + result[1]
            result = (metric, n)
        return result

    def extract_output(
        self, accumulator: Tuple[Dict[int, float], int]
    ) -> Dict[int, float]:
        return {k: v / accumulator[1] for k, v in accumulator[0].items()}


# Orderless Metrics
class _HitRatioTopKPreprocessor(TopKMetricPreprocessor):
    """Hit Ratio computation logic."""

    def accumulate_metric(self, element: Dict[str, Any]):
        """Generator function for beam.DoFn"""
        # dense tensor, int or str, (batch_size, None)
        y_pred: np.ndarray = element[self.prediction_key]
        # dense or sparse tensor, int or str
        y_label: Union[np.ndarray, sparray, List[List]] = (
            element[self.label_key]
            if self.label_key in element
            # Attempting to get from the feature
            else element[self.feature_key][self.label_key]
        )
        y_label, padding = self.sparse_to_dense(y_label, "")

        metrics = {}
        for k in self.top_k:
            pred = y_pred[:, :k]
            y_intersect = self.set_operation(
                y_label,
                pred,
                pad=padding,
                operation="intersection",
                returns="count",
            )
            metrics[k] = (y_intersect > 0).sum()
        yield metrics, len(y_pred)


class HitRatioTopK(BaseMetric):
    """
    Hit Ratio metric.
    If set size of intersection(label, pred) > 0, score as 1,
    otherwise, score as 0.
    When summarizing, take average for all samples.
    """

    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
    ):
        super(HitRatioTopK, self).__init__(
            name="hit_ratio",
            preprocessors=[
                _HitRatioTopKPreprocessor(
                    top_k=top_k,
                    feature_key=feature_key,
                    prediction_key=prediction_key,
                    label_key=label_key,
                    weight_key=weight_key,
                )
            ],
            combiner=SampleTopKMetricCombiner(metric_key="hit_ratio", top_k=top_k),
        )


# Ordered Metrics


class _NDCGTopKPreprocessor(TopKMetricPreprocessor):
    """NDCG computation logic."""

    def accumulate_metric(self, element: Dict[str, Any]):
        # dense tensor, int or str, (batch_size, None)
        y_pred: np.ndarray = element[self.prediction_key]
        # dense or sparse tensor, int or str
        y_label: Union[np.ndarray, sparray, List[List]] = (
            element[self.label_key]
            if self.label_key in element
            # Attempting to get from the feature
            else element[self.feature_key][self.label_key]
        )
        y_label, _ = self.sparse_to_dense(y_label, "")
        # for weighted NDCG
        y_weight: Union[np.ndarray, sparray, List[List], None] = (
            element[self.weight_key]
            if self.weight_key in element
            else element.get(self.feature_key, {}).get(self.weight_key, None)
        )
        if y_weight is not None:
            y_weight, _ = self.sparse_to_dense(y_weight, 0.0)

        metrics = {}
        for k in self.top_k:
            pred = y_pred[:, :k]
            indicator = pred[..., None] == y_label[:, None]
            if y_weight is None:  # binary score
                rel = indicator.any(2).astype(float)
            else:
                rel = (indicator * y_weight[:, None, :]).sum(2)
            dcg = (2**rel - 1) / np.log2(2 + np.arange(k)[None, :])
            dcg = dcg.sum(axis=1)
            # Ideal ordering
            ideal = np.sort(rel, axis=1)[:, ::-1]
            idcg = (2**ideal - 1) / np.log2(2 + np.arange(k)[None, :])
            idcg = idcg.sum(axis=1)
            # Compute final ndcg
            ndcg = dcg / np.where(idcg < 1e-6, 1.0, idcg)
            metrics[k] = ndcg.sum()  # accumulate

        yield metrics, len(y_pred)


class NDCGTopK(BaseMetric):
    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
    ):
        super(NDCGTopK, self).__init__(
            name="hit_ratio",
            preprocessors=[
                _NDCGTopKPreprocessor(
                    top_k=top_k,
                    feature_key=feature_key,
                    prediction_key=prediction_key,
                    label_key=label_key,
                    weight_key=weight_key,
                )
            ],
            combiner=SampleTopKMetricCombiner(metric_key="ndcg", top_k=top_k),
        )


# %% Population Level Metrics


class PopulationTopKMetricPreprocessor(TopKMetricPreprocessor):
    """Accumulate a Counter for categories/tokens in the predicion,
    along with other fields in the inputs.

    - other_fields: additional fields to accumulate the values on.
        Assuming this field/feature shares the same vocabulary set
        as the prediction.
    - vocabulary_fields: use a list of "{feature_name}" or "label",
        "prediction" to estimate the overall vocabulary as we iterate
        through the data. This is helpful when we do not know the
        vocabulary beforehand.
    """

    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
        group_keys: List[List[str]] = None,
        other_fields: List[str] = [],
        vocabulary_fields: List[str] = [],
    ):
        super(PopulationTopKMetricPreprocessor, self).__init__(
            top_k=top_k,
            feature_key=feature_key,
            prediction_key=prediction_key,
            label_key=label_key,
            weight_key=weight_key,
            group_keys=group_keys,
        )
        # Additional field for accumulation
        # the field shares the vocabulary set of the prediction
        # such as content interaction history feature
        self.other_fields = other_fields
        # If specified, use these fields to estimate the
        # vocabulary of the metric. Values can be
        # "label", "prediction", or "{feature_name}"
        self.vocabulary_fields = vocabulary_fields

    def accumulate_metric(self, element: Dict[str, Any]):
        results, _vocab = {}, set()
        y_pred: np.ndarray = element[self.prediction_key]
        if "prediction" in self.vocabulary_fields:
            _vocab.update(set(y_pred.ravel()))
        num = y_pred.shape[0]
        for k in self.top_k:
            results[k] = Counter(y_pred[:, :k].ravel())
        # Additional fields for accumulation
        if "label" in self.other_fields or "label" in self.vocabulary_fields:
            y_label: Union[np.ndarray, sparray, List[List]] = (
                element[self.label_key]
                if self.label_key in element
                # Attempting to get from the feature
                else element[self.feature_key][self.label_key]
            )
            y_label = self.flatten_array(y_label)
            if "label" in self.other_fields:
                results["label"] = Counter(y_label)
            if "label" in self.vocabulary_fields:
                _vocab.update(set(y_label))
        # Additional fields from features
        for field in set(self.other_fields + self.vocabulary_fields):
            if field == "label":
                continue  # skip label
            if field not in element.get(self.feature_key, {}):
                continue  # skip non-existent keys
            feature = element[self.feature_key][field]
            feature = self.flatten_array(feature)
            if field in self.other_fields:
                results[field] = Counter(feature)
            if field in self.vocabulary_fields:
                _vocab.update(set(feature))

        # Aggregate on the estimated vocab as well
        if self.vocabulary_fields:
            results["_vocab"] = Counter(_vocab)
        yield results, num


class PopulationTopKMetricCombiner(beam.CombineFn):
    def __init__(
        self,
        metric_key: str,
        top_k: Union[int, List[int]],
        other_fields: List[Union[str, int]] = [],
        vocabulary: Optional[List[str]] = None,
        retain_zeros: bool = False,
        accumulate_vocabulary: bool = False,
    ):
        """
        Population levle top-k metric combiner class.

        Args:
            metric_key (str): Name of the metric
            top_k (Union[int, List[int]]): Either integer or a list of integers for top_k
                metric calculation
            vocabulary (List[str], optional): A list of vocabulary for the counted class.
                When computing certain metrics, if the key is missing from the counter,
                but the key is present in vocabulary, the entry is recorded as zero.
                Conversely, if extra keys exist outside of the vocabulary, the keys are
                excluded from the calculation. Defaults to None
            retain_zeros (bool): retain the counter keys with 0 counts when updating with
                add_inputs or merge_accumulators. Default to False (faster).
            accumulate_vocabulary (bool): whether or not adding the "_vocab" field
                in the accumulator
        """
        super().__init__()
        self.metric_key = metric_key
        self.top_k = [top_k] if isinstance(top_k, int) else top_k
        self.other_fields = other_fields
        self.vocabulary = vocabulary
        self.retain_zeros = retain_zeros
        self.accumulate_vocabulary = accumulate_vocabulary

    def create_accumulator(self) -> Tuple[Dict[int, float], int]:
        # top-k accumulator, count
        return {
            k: Counter()
            for k in self.top_k
            + self.other_fields
            + (["_vocab"] if self.accumulate_vocabulary else [])
        }, 0

    def add_input(
        self,
        accumulator: Tuple[Dict[int, Counter], int],
        state: Tuple[Dict[int, Counter], int],
    ) -> Tuple[Dict[int, Counter], int]:
        if self.retain_zeros:
            metrics = {}
            for k in state[0]:
                metrics[k] = Counter(accumulator[0].get(k, Counter()))
                metrics[k].update(state[0][k])
        else:
            metrics = {
                k: accumulator[0].get(k, Counter()) + state[0][k] for k in state[0]
            }
        n = accumulator[1] + state[1]
        return metrics, n

    def merge_accumulators(
        self, accumulators: Iterable[Tuple[Dict[int, float], int]]
    ) -> Tuple[Dict[int, float], int]:
        accumulators = iter(accumulators)
        result = next(accumulators)  # get the first item
        for accumulator in accumulators:
            if self.retain_zeros:
                metrics = {}
                for k in accumulator[0]:
                    metrics[k] = Counter(result[0].get(k, Counter()))
                    metrics[k].update(accumulator[0][k])
            else:
                metrics = {
                    k: accumulator[0].get(k, Counter()) + result[0][k]
                    for k in result[0]
                }
            n = accumulator[1] + result[1]
            result = (metrics, n)
        return result

    def extract_output(
        self, accumulator: Tuple[Dict[int, float], int]
    ) -> Dict[int, float]:
        """Needs to be implemented for specific metrics."""
        return accumulator


class _CoverageTopKCombiner(PopulationTopKMetricCombiner):
    def extract_output(
        self, accumulator: Tuple[Dict[int, float], int]
    ) -> Dict[int, float]:
        accumulator, _ = accumulator
        if self.vocabulary:
            vocabulary = list(set(self.vocabulary))
        else:  # assume the accumulator has all the vocabularies
            vocabulary = accumulator.pop("_vocab")

        # Compute coverage
        coverage = {}
        for k in accumulator:
            covered = accumulator[k].keys()
            covered = set(covered).intersection(set(vocabulary))
            coverage[k] = len(covered) / len(vocabulary)
        return coverage


class CoverageTopK(BaseMetric):
    """Coverage metric."""

    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
        include_labels: bool = True,
        vocabulary: Optional[Union[List[str], None]] = None,
        vocabulary_fields: List[str] = [],
    ):
        """
        Args:
            top_k (Union[int, List[int]]): Top-k of predictions
                used to compute coverage.
            include_labels (bool, optional): Whehter or not
                to include coverage metric for labels.
                Defaults to True.
            vocabulary (List[str]): Explicit list of vocabulary
                to use when calculating coverage metrics.
            vocabulary_fields (List[str]): Data feature fields,
                or "label", or "prediction" that can be used to
                estimate the list of vocabulary. Ignored when
                vocabulary is set. Otherwise, either explicitly
                set the columns or "prediction" will be used
                to estimate the vocabulary. Note that this is
                slightly more computationally expensive.
        """
        top_k = [top_k] if isinstance(top_k, int) else top_k
        vocabulary_fields = (
            (vocabulary_fields or ["prediction"]) if not vocabulary else []
        )
        super(CoverageTopK, self).__init__(
            name="coverage",
            preprocessors=[
                PopulationTopKMetricPreprocessor(
                    top_k=top_k,
                    feature_key=feature_key,
                    prediction_key=prediction_key,
                    label_key=label_key,
                    weight_key=weight_key,
                    other_fields=["label"] if include_labels else [],
                    vocabulary_fields=vocabulary_fields,
                )
            ],
            combiner=_CoverageTopKCombiner(
                metric_key="coverage",
                top_k=top_k,
                vocabulary=vocabulary,
                accumulate_vocabulary=True if not vocabulary else False,
            ),
        )


class _RedundancyTopKCombiner(PopulationTopKMetricCombiner):
    def extract_output(
        self, accumulator: Tuple[Dict[int, float], int]
    ) -> Dict[int, float]:
        accumulator, num = accumulator

        # Compute coverage
        coverage = {}
        for k in accumulator:
            try:
                k = int(k)  # assuming k is top_k integer
            except:
                continue
            covered = accumulator[k].keys()
            coverage[k] = 1 - len(covered) / (num * k)
        return coverage


class RedundacyTopK(BaseMetric):
    """Redundancy metric.
    Given top-k recommendations, there should be
    num_examples x top_k available slots. Redundancy
    is then calcualted as

    1 - num_unique_items / (num_examples x top_k)

    The division term computes the proportion of
    slots that contains unique items. The higher this
    number, the more redundancy there is in the
    recommenation.
    """

    def __init__(
        self,
        top_k: Union[int, List[int]],
        prediction_key: str = DEFAULT_PREDICTION_KEY,
    ):
        """
        Args:
            top_k (Union[int, List[int]]): Top-k of predictions
                used to compute coverage.
        """
        top_k = [top_k] if isinstance(top_k, int) else top_k
        super(RedundacyTopK, self).__init__(
            name="redundancy",
            preprocessors=[
                PopulationTopKMetricPreprocessor(
                    top_k=top_k,
                    prediction_key=prediction_key,
                    other_fields=[],
                    vocabulary_fields=[],
                )
            ],
            combiner=_RedundancyTopKCombiner(
                metric_key="redundancy",
                top_k=top_k,
            ),
        )


# %% Approximate Count Metrics


class _UniqueCountTopKPreprocessor(TopKMetricPreprocessor):
    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
        group_keys: List[List[str]] = None,
        include_labels: bool = True,
        use_ordered_list: bool = True,
    ):
        super(_UniqueCountTopKPreprocessor, self).__init__(
            top_k=top_k,
            feature_key=feature_key,
            prediction_key=prediction_key,
            label_key=label_key,
            weight_key=weight_key,
            group_keys=group_keys,
        )
        self.include_labels = include_labels
        # if true, treat the recommendation as an ordered list
        # so that even if the same set of contents are recommended
        # if the ordering of them are different, they will be treated
        # as a different pattern
        self.use_ordered_list = use_ordered_list

    def accumulate_metric(self, element):
        results = {}
        y_pred: np.ndarray = element[self.prediction_key]
        for k in self.top_k:
            pred = y_pred[:, :k]
            if not self.use_ordered_list:
                pred = np.sort(pred, axis=1)
            pred = pd.Series(pred.tolist()).str.join("")
            # initial round of dedup
            results[str(k)] = pred.unique().tolist()

        # Process labels
        if self.include_labels:
            y_label: Union[np.ndarray, sparray, List[List]] = (
                element[self.label_key]
                if self.label_key in element
                # Attempting to get from the feature
                else element[self.feature_key][self.label_key]
            )
            y_label, _ = self.sparse_to_dense(y_label, pad="")
            y_label = pd.Series(y_label.tolist()).str.join("")
            # initial round of dedup
            results["label"] = y_label.unique().tolist()
        
        # emitting one result at a time for approximatecountcombiner
        for k, v in results.items():
            for vv in v:
                yield {k: vv}


class UniqueCountTopK(BaseMetric):
    def __init__(
        self,
        top_k: Union[int, List[int]],
        feature_key: str = DEFAULT_FEATURE_KEY,
        prediction_key: str = DEFAULT_PREDICTION_KEY,
        label_key: str = DEFAULT_LABEL_KEY,
        weight_key: str = None,
        error: float = 0.01,
        include_labels: bool = True,
        use_ordered_list: bool = True,
    ):
        _preprocessor = _UniqueCountTopKPreprocessor(
            top_k=top_k,
            feature_key=feature_key,
            prediction_key=prediction_key,
            label_key=label_key,
            weight_key=weight_key,
            include_labels=include_labels,
            use_ordered_list=use_ordered_list,
        )
        self._sample_size = ApproximateUnique.parse_input_params(None, error)

        super(UniqueCountTopK, self).__init__(
            name="unique_count",
            preprocessors=[_preprocessor],
            # one combiner per output key
            combiner={
                str(k): ApproximateUniqueCombineFn(
                    self._sample_size, coder=beam.coders.StrUtf8Coder()
                )
                for k in _preprocessor.top_k + (["label"] if include_labels else [])
            },
        )
