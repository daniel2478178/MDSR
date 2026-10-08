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
import ast
import re
MAX_RETRIES =3
llm_url="api.deepseek.com"
verbose = False
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
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
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
            "Prefer equations using no more than 4 dataset-specific parameters unless additional parameters are clearly necessary.\n"
            "Requirements:\n"
            "- Keep exactly the same function name.\n"
            "- Keep exactly the same function arguments.\n"
            "- Improve or replace the mathematical expression in the body.\n"
            "- Use params[i] as free numerical parameters when appropriate.\n"
            "- Use only standard Python math operators and functions.\n"
            "- The function MUST contain executable Python code.\n"
            "- The function MUST contain an explicit return statement.\n"
            "- NEVER return a docstring-only function.\n"
            "- NEVER return an empty function.\n"
            "- Do not include explanations.\n"
            "- Do not include Markdown code fences.\n"
            "The function MUST be named exactly `equation`."
            "Always return:"
            "def equation("
            "    prob: np.ndarray,"
            "    t: np.ndarray,"
            "    omega: np.ndarray,"
            "    params: np.ndarray,"
            ") -> np.ndarray:"
            "    ..."
            "    return y"

            "Never rename the function to equation_v1, equation_v2, etc."

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

                    print("\n" + "=" * 100)
                    print("FULL API RESPONSE")
                    print("=" * 100)

                    try:
                        print(
                            json.dumps(
                                data,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                    except Exception:
                        print(data)

                    print("=" * 100)

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

                    print("\n" + "=" * 100)
                    print("FINAL SAMPLE SENT TO EVALUATOR")
                    print("=" * 100)
                    print(repr(response))
                    print("=" * 100)

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
        print("\n" + "=" * 100)
        print("PROMPT SENT TO LLM")
        print("=" * 100)
        print(data)
        print("=" * 100 + "\n")
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        print("\n" + "=" * 100)
        print("RAW LLM RESPONSE")
        print("=" * 100)
        print(response)
        print("=" * 100 + "\n")

        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]

