from itertools import product
import pandas as pd
import multiprocessing as mp
import numpy as np


def product_dict(**kwargs):
    """
    Usage:
        n = [500,1000,2000]
        p = [1000,2000]

        def test(n=1,p=1):
            print(n,p)

        for instance in product_dict(n=n,p=p):
            test(**instance)
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def merge_dict(list_of_dict1, list_of_dict2):
    """
    Usage:
        d1 = [{'n': 1, 'm':2},{'n': 11, 'm': 12}]
        d2 = [{'a':0.9,'fpr': 0.1}, {'a':0.92,'fpr': 0.14}]
        r = []
        r.extend(merge_dict(d1, d2))
        print(r)
    """
    for d in zip(list_of_dict1, list_of_dict2):
        d[0].update(d[1])
        yield d[0]

def del_duplicate(*lists_of_dict):
    """
    >>>list(del_duplicate([{"a": 1}, {"b": 2}, {"a": 1}], [{"a": 2}, {"a": 1}]))
    >>>[{'b': 2}, {'a': 1}, {'a': 2}]
    """
    for tuple_dict in set(
        [tuple(dict.items()) for list_of_dict in lists_of_dict for dict in list_of_dict]
    ):
        yield dict(tuple_dict)

def accuracy(model_coef, data_coef):
    """
    use fo variables selection
    """
    model_coef = set(np.nonzero(model_coef)[0])
    data_coef = set(np.nonzero(data_coef)[0])
    return len(model_coef & data_coef) / len(data_coef)

def FDR(model_coef, data_coef):
    """
    use fo variables selection
    """
    model_coef = set(np.nonzero(model_coef)[0])
    data_coef = set(np.nonzero(data_coef)[0])
    return len(model_coef - data_coef) / len(model_coef) if len(model_coef) > 0 else 0

class Test:
    def __init__(self, task, in_keys, out_keys, processes=1, name="test"):
        """
        in_keys:
            in_keys are arrays of strings which will be the keys of task's in_para.
        out_keys:
            out_keys are arrays of strings which will be the keys of task's out_para.
        """
        self.task = task
        self.in_keys = in_keys
        self.out_keys = out_keys
        assert len(set(self.in_keys) & set(self.out_keys)) == 0
        self.name = name
        self.results = []  # 'results' is a list of dict
        self.processes = processes

    def task_parallel(self, in_para):
        try:
            result = self.task(**in_para)
        except Exception as e:
            print(e)
            result = {para: np.nan for para in self.out_keys}
        in_para.update(result)
        print(in_para, flush=True)
        return in_para

    def check(self, **in_para):
        if set(in_para.keys()) != set(self.in_keys):
            raise RuntimeError("in_parameter's keys do not match!\n{}\n{}".format(self.in_keys, set(in_para.keys())))

        result = self.task(**in_para)
        print(result)

        if not isinstance(result, dict) or set(result.keys()) != set(self.out_keys):
            raise RuntimeError("out_parameter's keys do not match!\n{}\n{}".format(self.out_keys, set(result.keys())))

    def run(self, in_para_list = None, /, **in_para):
        """
        in_para.keys should be the same as self.in_keys and in_para.values should be arrays.
        """
        if in_para_list is None:
            in_para_list = product_dict(**in_para)

        with mp.Pool(processes=self.processes) as pool:
            self.results.extend(
                pool.starmap(
                    self.task_parallel, [(para,) for para in in_para_list]
                ),
            )

    def save(self, filename=None):
        if filename is None:
            filename = self.name

        pd.DataFrame(
            {
                para: [result[para] for result in self.results]
                for para in self.in_keys + self.out_keys
            }
        ).to_csv(filename + ".csv")



if __name__ == "__main__":
    #mp.set_start_method('spawn')
    import time
    in_keys= ['n','p','k']
    out_keys= ['seed', 'time', 'ac']
    def f(n,p,k):
        time.sleep(1.0)
        if n==3:
            raise RuntimeError('error')
        return {'seed': n+100, 'time': p+100, 'ac': k/(k+1)}
    test = Test(f,in_keys,out_keys, processes=10)
    test.check(n=1,p=3,k=2)
    test.run(n=[11,2,3],p=[5,6],k=[7,8])
    test.save()
