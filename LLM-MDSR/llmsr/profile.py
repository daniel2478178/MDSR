# profile the experiment with tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from llmsr import code_manipulation
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

        # Only log a text summary if there is a valid best function to report.
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )
    def _write_json(self, program):
        sample_order = getattr(program, "global_sample_nums", None) or 0

        nmse_per_test = getattr(program, "nmse_per_test", None)
        if isinstance(nmse_per_test, dict):
            nmse_per_test = {str(k): float(v) for k, v in nmse_per_test.items()}

        content = {
            "sample_order": sample_order,
            "function": str(program),
            "score": getattr(program, "score", None),
            "complexity": getattr(program, "complexity", None),
            "mean_nmse": getattr(program, "mean_nmse", None),
            "min_nmse": getattr(program, "min_nmse", None),
            "max_nmse": getattr(program, "max_nmse", None),
            "median_nmse": getattr(program, "median_nmse", None),
            "std_nmse": getattr(program, "std_nmse", None),
            "worst2_mean_nmse": getattr(program, "worst2_mean_nmse", None),
            "nmse_per_test": nmse_per_test,
            "sample_time": getattr(program, "sample_time", None),
            "evaluate_time": getattr(program, "evaluate_time", None),
        }

        path = os.path.join(self._json_dir, f"samples_{sample_order}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    # def _write_json(self, programs: code_manipulation.Function):
    #     sample_order = programs.global_sample_nums
    #     sample_order = sample_order if sample_order is not None else 0
    #     function_str = str(programs)
    #     score = programs.score
    #     content = {
    #         'sample_order': sample_order,
    #         'function': function_str,
    #         'score': score
    #     }
    #     path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
    #     with open(path, 'w') as json_file:
    #         json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        # print(f'================= Evaluated Function =================')
        # print(f'{function_str}')
        # print(f'------------------------------------------------------')
        # print(f'Score        : {str(score)}')
        # print(f"Mean NMSE          : {getattr(function, 'mean_nmse', None)}")
        # print(f"Complexity         : {getattr(function, 'complexity', None)}")
        # print(f"Min NMSE           : {getattr(function, 'min_nmse', None)}")
        # print(f"Max NMSE           : {getattr(function, 'max_nmse', None)}")
        # print(f"Median NMSE        : {getattr(function, 'median_nmse', None)}")
        # print(f"Std NMSE           : {getattr(function, 'std_nmse', None)}")
        # print(f"Worst-2 Mean NMSE  : {getattr(function, 'worst2_mean_nmse', None)}")
        # print(f'Sample time  : {str(sample_time)}')
        # print(f'Evaluate time: {str(evaluate_time)}')
        # print(f'Sample orders: {str(sample_orders)}')
        # print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time
        if evaluate_time is not None:
            self._tot_evaluate_time += evaluate_time
