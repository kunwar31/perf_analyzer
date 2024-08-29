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


from argparse import Namespace

from genai_perf.export_data.data_exporter_factory import DataExporterFactory
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.metrics import Statistics, TelemetryMetricsStatistics
from genai_perf.parser import get_extra_inputs_as_dict


class OutputReporter:
    """
    A class to orchestrate output generation.
    """

    def __init__(
        self,
        stats: Statistics,
        telemetry_stats: TelemetryMetricsStatistics,
        args: Namespace,
    ):
        self.args = args
        self.stats = stats
        self.telemetry_stats = telemetry_stats
        self.stats.scale_data()

    def report_output(self) -> None:
        factory = DataExporterFactory()
        exporter_config = self._create_exporter_config()
        data_exporters = factory.create_data_exporters(exporter_config)

        for exporter in data_exporters:
            exporter.export()

    def _create_exporter_config(self) -> ExporterConfig:
        config = ExporterConfig()
        config.stats = self.stats.stats_dict
        if self.telemetry_stats is not None:
            config.telemetry_stats = self.telemetry_stats.stats_dict
        config.metrics = self.stats.metrics
        config.args = self.args
        config.artifact_dir = self.args.artifact_dir
        config.extra_inputs = get_extra_inputs_as_dict(self.args)

        return config
