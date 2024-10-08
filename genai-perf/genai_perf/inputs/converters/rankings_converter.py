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

from typing import Any, Dict

from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.inputs_config import InputsConfig


class RankingsConverter(BaseConverter):

    def convert(self, generic_dataset: Dict, config: InputsConfig) -> Dict:
        if "queries" not in generic_dataset or "passages" not in generic_dataset:
            raise ValueError(
                "Both 'queries.jsonl' and 'passages.jsonl' must be present in the input datasets."
            )

        passages_dataset = generic_dataset["passages"]
        queries_dataset = generic_dataset["queries"]
        request_body: Dict[str, Any] = {"data": []}

        if not queries_dataset["rows"]:
            raise ValueError("'queries.jsonl' must contain at least one entry.")
        if not passages_dataset["rows"]:
            raise ValueError("'passages.jsonl' must contain at least one entry.")

        print("queries_dataset", queries_dataset)
        print("passages_dataset", passages_dataset)
        rows_of_passage_data = len(passages_dataset["rows"])
        for query_index, query_entry in enumerate(queries_dataset["rows"]):
            if query_index > rows_of_passage_data:
                break

            model_name = self._select_model_name(config, query_index)
            print("type(query_entry)", type(query_entry))
            print("query_entry", query_entry)
            print("type(query_entry(row))", type(query_entry["text_input"]))
            print("query_entry(text_input)", query_entry["text_input"])
            query = query_entry["text_input"]

            passage_entry = passages_dataset["rows"][query_index]

            if self._is_rankings_tei(config):
                passages = [
                    p.get("text_input")
                    for p in passage_entry
                    if p.get("text_input") is not None
                ]
                payload = {"query": query, "texts": passages}
            else:
                passages = [
                    {"text_input": p.get("text_input")}
                    for p in passage_entry
                    if p.get("text_input") is not None
                ]
                payload = {
                    "query": query,
                    "passages": passages,
                    "model": model_name,
                }

            self._add_request_params(payload, config)
            request_body["data"].append({"payload": [payload]})

        return request_body

    def _is_rankings_tei(self, config: InputsConfig) -> bool:
        """
        Check if user specified that they are using the Hugging Face
        Text Embeddings Interface for ranking models
        """
        if config.extra_inputs.get("rankings") == "tei":
            return True
        return False

    def _add_request_params(self, payload: Dict, config: InputsConfig) -> None:
        for key, value in config.extra_inputs.items():
            if not (key == "rankings" and value == "tei"):
                payload[key] = value
