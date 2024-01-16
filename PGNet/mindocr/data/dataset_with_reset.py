import builtins

from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import BatchDataset
from mindspore.dataset.engine.validators import check_batch


class BatchDatasetWithReset(BatchDataset):
    def reset(self):
        for item in self.children:
            if hasattr(item, "reset"):
                item.reset()


class GeneratorDatasetWithReset(GeneratorDataset):
    def reset(self):
        """Reset the dataset for next epoch."""
        if isinstance(self.source, builtins.zip):
            # Although zip is iteratable, it does not have the feature of repeated iteration, so pass it to the array.
            for item in self.source:
                if getattr(item, "need_reset", False) and hasattr(item, "reset"):
                    item.reset()
        else:
            if getattr(self.source, "need_reset", False) and hasattr(self.source, "reset"):
                self.source.reset()

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, **kwargs):
        """
        Combine batch_size number of consecutive rows into batch which apply per_batch_map to the samples first.

        For any column, all the elements within that column must have the same shape.

        Refer to the following figure for the execution process:

        .. image:: batch_en.png

        Note:
            The order of using repeat and batch reflects the number of batches and per_batch_map.
            It is recommended that the repeat operation applied after the batch operation finished.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: False. If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: None.
            **kwargs:

                - per_batch_map (Callable[[List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo], \
                  (List[numpy.ndarray], ..., List[numpy.ndarray])], optional): Per batch map callable. Default: None.
                  A callable which takes (List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo) as input parameters.
                  Each list[numpy.ndarray] represents a batch of numpy.ndarray on a given column. The number of lists
                  should match with the number of entries in input_columns. The last parameter of the callable should
                  always be a BatchInfo object. Per_batch_map should return
                  (list[numpy.ndarray], list[numpy.ndarray], ...). The length of each list in output should be the same
                  as the input. output_columns is required if the number of output lists is different from input.

                - input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of
                  the list should match with signature of per_batch_map callable. Default: None.

                - output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                  outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                  len(output_columns). The size of this list must match the number of output
                  columns of the last operation. Default: None, output columns will have the same
                  name as the input columns, i.e., the columns will be replaced.

                - python_multiprocessing (bool, optional): Parallelize Python function `per_batch_map` with
                  multi-processing or multi-threading mode, True means multi-processing, False means multi-threading
                  If `per_batch_map` is a I/O bound task, use multi-threading mode.
                  If `per_batch_map` is a CPU bound task, it is recommended to use multi-processing mode.
                  Default: False, use python multi-threading mode.

                - max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to
                  copy data between processes. This is only used if python_multiprocessing is set to True. Default: 16.

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> # 1) Create a dataset where every 100 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.batch(100, True)
            >>>
            >>> # 2) resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
            >>> def np_resize(col, BatchInfo):
            ...     output = col.copy()
            ...     s = (BatchInfo.get_batch_num() + 1) ** 2
            ...     index = 0
            ...     for c in col:
            ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
            ...         img = img.resize((s, s))
            ...         output[index] = np.array(img)
            ...         index += 1
            ...     return (output,)
            >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)
            >>>
            >>> # 3) Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each time.
            >>> def add_one(BatchInfo):
            ...     return BatchInfo.get_batch_num() + 1
            >>> dataset = dataset.batch(batch_size=add_one, drop_remainder=True)
        """
        return BatchDatasetWithReset(self, batch_size, drop_remainder, num_parallel_workers, **kwargs)
