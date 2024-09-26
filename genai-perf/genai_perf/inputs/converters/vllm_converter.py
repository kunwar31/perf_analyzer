# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import random
from typing import Any, Dict

from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import DEFAULT_OUTPUT_TOKENS_MEAN
from genai_perf.inputs.inputs_config import InputsConfig


class VLLMConverter(BaseConverter):

    _CONTENT_NAMES = [
        "text_input",
        # OPENORCA
        "system_prompt",
        "question",
        # CNN DAILYMAIL
        "article",
    ]

    def convert(self, generic_dataset: Dict, config: InputsConfig) -> Dict:
        """
        Construct a request body using the endpoint specific request format.
        """
        request_body: Dict[str, Any] = {"data": []}

        for index, entry in enumerate(generic_dataset["rows"]):
            model_name = self._select_model_name(config, index)
            text_input = self._construct_text_payload(entry)

            payload = {
                "model": model_name,
                "text_input": text_input,
                "exclude_input_in_output": [True],  # default
            }
            self._add_request_params(payload, config)
            request_body["data"].append(payload)

        return request_body

    def _add_request_params(self, payload: Dict, config: InputsConfig) -> None:
        if config.add_stream:
            payload["stream"] = [True]
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = str(
                int(
                    max(
                        0,
                        random.gauss(
                            config.output_tokens_mean,
                            config.output_tokens_stddev,
                        ),
                    )
                )
            )
            sampling_parameters = {
                "max_tokens": number_of_tokens,
            }
            if config.output_tokens_deterministic:
                sampling_parameters["min_tokens"] = number_of_tokens
            sampling_parameters_str = json.dumps(sampling_parameters)
            payload["sampling_parameters"] = [sampling_parameters_str]
        for key, value in config.extra_inputs.items():
            payload[key] = [value]
