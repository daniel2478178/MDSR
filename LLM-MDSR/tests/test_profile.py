import tempfile

from llmsr.code_manipulation import Function
from llmsr.profile import Profiler


def test_register_function_without_valid_score_does_not_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(log_dir=tmpdir)
        function = Function(name='test_func', args='x', body='return x')
        function.global_sample_nums = 1
        function.sample_time = 0.5
        function.evaluate_time = 0.2
        function.score = None

        profiler.register_function(function)

        # This should not raise when there is no valid best-function score yet.
        profiler._write_tensorboard()
