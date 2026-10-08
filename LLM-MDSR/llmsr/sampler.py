# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib
import requests
import json
import http.client
import os
import pickle
import ast
import re
MAX_RETRIES =3
llm_url="api.deepseek.com"
verbose = False
MAX_FORMULA_COMPLEXITY = 25

# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get(
    "LLMSR_CHECKPOINT_PATH",
    os.path.join("logs", "llmsr_checkpoint.pkl"),
)
CHECKPOINT_INTERVAL = 10


def save_checkpoint(path, database, global_sample_nums):
    """Atomically save the LLMSR search state."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "database": database,
        "global_sample_nums": int(global_sample_nums),
        "numpy_random_state": np.random.get_state(),
    }

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def load_checkpoint(path):
    """Load a checkpoint if it exists, otherwise return None."""
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 0 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            start_sample_num: int = 0,
            checkpoint_path: str | None = CHECKPOINT_PATH,
            checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        if start_sample_num is None:
            start_sample_num = 0
        self.set_global_sample_nums(start_sample_num)
        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval = checkpoint_interval

    
    def sample(self, **kwargs):
        """Continuously get prompts, sample programs, and send them for analysis."""
        while True:
            if (
                self._max_sample_nums is not None
                and self._get_global_sample_nums() >= self._max_sample_nums
            ):
                self._save_checkpoint_if_needed(force=True)
                break

            prompt = self._database.get_prompt()

            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / max(len(samples), 1)

            for sample in samples:
                # A returned batch can otherwise overshoot max_sample_nums.
                if (
                    self._max_sample_nums is not None
                    and self._get_global_sample_nums() >= self._max_sample_nums
                ):
                    self._save_checkpoint_if_needed(force=True)
                    return

                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()

                chosen_evaluator: evaluator.Evaluator = np.random.choice(
                    self._evaluators
                )
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time,
                )

                # Save only after analyse() succeeds, so the counter and
                # ExperienceBuffer in the checkpoint describe the same state.
                self._save_checkpoint_if_needed()

    def _save_checkpoint_if_needed(self, force: bool = False):
        if self._checkpoint_path is None:
            return

        cur = self._get_global_sample_nums()

        if force or (
            self._checkpoint_interval > 0
            and cur % self._checkpoint_interval == 0
        ):
            save_checkpoint(
                self._checkpoint_path,
                self._database,
                cur,
            )
            print(
                f"[checkpoint] saved at sample {cur}: "
                f"{self._checkpoint_path}"
            )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt = (
            "You are a symbolic regression assistant discovering mathematical "
            "laws for scientific systems.\n\n"

            "Several versions of a mathematical function are provided. "
            "The LAST versioned function is the function that you must improve.\n\n"

            "Rewrite the mathematical body of that final function and return "
            "exactly ONE COMPLETE Python function definition.\n\n"

            "Prefer equations using no more than 4 dataset-specific parameters "
            "unless additional parameters are clearly necessary.\n\n"

            "Requirements:\n"
            "- The formula complexity must not exceed "
            f"{MAX_FORMULA_COMPLEXITY} mathematical operations.\n"
            "- Copy the function name exactly from the LAST function in the prompt.\n"
            "- Copy the complete function signature exactly, including all arguments "
            "and type annotations.\n"
            "- Do not use the name or signature of an earlier function version.\n"
            "- Improve or replace the mathematical expression in the body.\n"
            "- Use params[i] as free numerical parameters when appropriate.\n"
            "- Use only NumPy-compatible mathematical operators and functions.\n"
            "- The function must contain executable Python code.\n"
            "- The function must contain an explicit return statement.\n"
            "- Never return a docstring-only or empty function.\n"
            "- Never use undefined variables.\n"
            "- Do not include explanations or Markdown code fences.\n"
            "- Do not use control-flow constructs inside the function: "
            "if/elif/else, conditional expressions, for or while loops, "
            "comprehensions, try/except/finally, with, match/case, "
            "break, continue, or raise.\n\n"
            "- Division and reciprocal relationships are allowed and should be"
            "considered when physically meaningful.\n"
            "- Independent variables may appear in the denominator.\n"
            "- Do not avoid division merely because it may introduce singularities,"
            " the supplied data already satisfy the stated domain restrictions.\n"        
            "Output format:\n"
            "- Output exactly one complete Python function.\n"
            "- Use the name and complete signature of the LAST function in the prompt.\n"
            "- Put the function body on lines following the function header.\n"
            "- End the function with an explicit return statement.\n"
        )
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)




    def extract_llm_code(
            self,
            data,
            verbose: bool = False,
    ) -> str:
        """
        Extract assistant content from an OpenAI-compatible API response.

        This function only normalizes the response and removes optional
        Markdown code fences. It intentionally does NOT require a specific
        function name, because LLMSR may generate equation_v1, equation_v2,
        etc., or return only a function body.
        """

        if not isinstance(data, dict):
            raise ValueError(
                f"API response is not a dict: {type(data).__name__}"
            )

        choices = data.get("choices")

        if not choices:
            raise ValueError(
                f"API response contains no choices: {data}"
            )

        choice = choices[0]

        finish_reason = choice.get("finish_reason")

        message = choice.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                f"API response contains invalid message: {message}"
            )

        content = message.get("content")

        if verbose:
            print("\n" + "=" * 100)
            print("LLM RESPONSE")
            print("=" * 100)
            print(f"finish_reason : {finish_reason}")
            print(f"content repr  : {repr(content)}")
            print("=" * 100)

        if content is None:
            raise ValueError(
                f"LLM returned content=None, "
                f"finish_reason={finish_reason}"
            )

        if not isinstance(content, str):
            raise ValueError(
                f"LLM content is not str: {type(content).__name__}"
            )

        content = content.strip()

        if not content:
            raise ValueError(
                f"LLM returned empty content, "
                f"finish_reason={finish_reason}"
            )

        # ------------------------------------------------------------
        # Remove optional Markdown code fence
        #
        # Supports:
        #
        # ```python
        # def equation(...):
        #     ...
        # ```
        #
        # and:
        #
        # ```
        # ...
        # ```
        # ------------------------------------------------------------

        blocks = re.findall(
            r"```(?:python)?\s*(.*?)```",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if blocks:
            content = blocks[0].strip()

        if verbose:
            print("\n" + "=" * 100)
            print("NORMALIZED LLM CONTENT")
            print("=" * 100)
            print(content)
            print("=" * 100)

        return content   

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
            # instruction
            prompt = '\n'.join([self._instruction_prompt, prompt])
            while True:
                try:
                    all_samples = []
                    # response from llm server
                    if self._batch_inference:
                        response = self._do_request(prompt)
                        for res in response:
                            all_samples.append(res)
                    else:
                        for _ in range(self._samples_per_prompt):
                            response = self._do_request(prompt)
                            all_samples.append(response)

                    # trim equation program skeleton body from samples
                    if self._trim:
                        all_samples = [_extract_body(sample, config) for sample in all_samples]
                    
                    return all_samples
                except Exception:
                    continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for sample_idx in range(self._samples_per_prompt):
            for attempt in range(MAX_RETRIES):
                try:
                    conn = http.client.HTTPSConnection(llm_url)
                    payload = json.dumps({
                        "max_tokens": 8092,
                        "model": config.api_model,
                        "thinking": {
                            "type": "disabled"
                         },

                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }


                    conn.request("POST", "/v1/chat/completions", payload, headers)

                    res = conn.getresponse()
                    # print(
                    #     json.dumps(
                    #         data,
                    #     indent=2,
                    #     ensure_ascii=False,
                    # )
                    #)
                    raw_response = res.read().decode("utf-8")

                    # ------------------------------------------------
                    # HTTP status check
                    # ------------------------------------------------

                    if res.status < 200 or res.status >= 300:
                        raise RuntimeError(
                            f"HTTP {res.status} "
                            f"{res.reason}\n"
                            f"{raw_response}"
                        )

                    # ------------------------------------------------
                    # Parse JSON
                    # ------------------------------------------------

                    data = json.loads(raw_response)

                    # ------------------------------------------------
                    # Debug full response
                    # ------------------------------------------------

                    # print("\n" + "=" * 100)
                    # print("FULL API RESPONSE")
                    # print("=" * 100)


                    # ------------------------------------------------
                    # Extract actual model content
                    # ------------------------------------------------

                    response = self.extract_llm_code(
                        data,
                        verbose,
                    )

                    # ------------------------------------------------
                    # LLMSR's original body extraction
                    # ------------------------------------------------

                    if self._trim:
                        response = _extract_body(
                            response,
                            config
                        )

                    # ------------------------------------------------
                    # Never allow empty sample
                    # ------------------------------------------------

                    if response is None:
                        raise ValueError(
                            "LLM response became None "
                            "after _extract_body()."
                        )

                    if not isinstance(response, str):
                        raise ValueError(
                            "LLM response after "
                            "_extract_body() is not str: "
                            f"{type(response).__name__}"
                        )

                    if not response.strip():
                        raise ValueError(
                            "LLM response became empty "
                            "after _extract_body()."
                        )

                    # print("\n" + "=" * 100)
                    # print("FINAL SAMPLE SENT TO EVALUATOR")
                    # print("=" * 100)
                    # print(repr(response))
                    # print("=" * 100)

                    all_samples.append(response)

                    break

                except Exception as e:

                    # CRITICAL:
                    # Never silently swallow exceptions.
                    print("\n" + "!" * 100)
                    print("LLM API / RESPONSE ERROR")
                    print("!" * 100)

                    print(
                        f"sample_idx : {sample_idx}"
                    )

                    print(
                        f"error type : "
                        f"{type(e).__name__}"
                    )

                    print(
                        f"error      : {e}"
                    )

                    print("!" * 100)

                    continue

                finally:

                    try:
                        conn.close()
                    except Exception:
                        pass        
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)


        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]

