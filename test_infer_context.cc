// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "client_backend/mock_client_backend.h"
#include "doctest.h"
#include "gmock/gmock.h"
#include "infer_context.h"
#include "mock_data_loader.h"
#include "mock_infer_context.h"
#include "mock_infer_data_manager.h"
#include "mock_sequence_manager.h"

namespace triton { namespace perfanalyzer {

/// Tests the round robin ordering of json input data
///
TEST_CASE("update_seq_json_data: testing the UpdateSeqJsonData function")
{
  std::shared_ptr<MockSequenceManager> mock_sequence_manager{
      std::make_shared<MockSequenceManager>()};

  EXPECT_CALL(
      *mock_sequence_manager, SetInferSequenceOptions(testing::_, testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return());

  mock_sequence_manager->InitSequenceStatuses(1);

  std::shared_ptr<MockDataLoader> mock_data_loader{
      std::make_shared<MockDataLoader>()};

  EXPECT_CALL(*mock_data_loader, GetTotalSteps(testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return(3));

  std::shared_ptr<MockInferDataManager> mock_infer_data_manager{
      std::make_shared<MockInferDataManager>()};

  testing::Sequence seq;
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 0, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 1, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 2, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 0, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 1, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 2, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));

  std::shared_ptr<MockInferContext> mic{std::make_shared<MockInferContext>()};

  EXPECT_CALL(*mic, SendRequest(testing::_, testing::_, testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return());

  mic->sequence_manager_ = mock_sequence_manager;
  mic->data_loader_ = mock_data_loader;
  mic->infer_data_manager_ = mock_infer_data_manager;
  mic->thread_stat_ = std::make_shared<ThreadStat>();
  bool execute{true};
  mic->execute_ = execute;
  mic->using_json_data_ = true;

  size_t seq_stat_index{0};
  bool delayed{false};

  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);

  // Destruct gmock objects to determine gmock-related test failure
  mock_sequence_manager.reset();
  mock_data_loader.reset();
  mock_infer_data_manager.reset();
  mic.reset();
  REQUIRE(testing::Test::HasFailure() == false);
}

TEST_CASE("send_request: testing the SendRequest function")
{
  MockInferContext mock_infer_context{};

  SUBCASE("testing logic relevant to request record sequence ID")
  {
    mock_infer_context.thread_stat_ = std::make_shared<ThreadStat>();
    mock_infer_context.thread_stat_->contexts_stat_.emplace_back();
    mock_infer_context.async_ = true;
    mock_infer_context.streaming_ = true;
    mock_infer_context.infer_data_.options_ =
        std::make_unique<cb::InferOptions>("my_model");
    std::shared_ptr<cb::MockClientStats> mock_client_stats{
        std::make_shared<cb::MockClientStats>()};
    mock_infer_context.infer_backend_ =
        std::make_unique<cb::MockClientBackend>(mock_client_stats);

    const uint64_t request_id{5};
    const bool delayed{false};
    const uint64_t sequence_id{2};

    mock_infer_context.infer_data_.options_->request_id_ =
        std::to_string(request_id);

    cb::MockInferResult* mock_infer_result{
        new cb::MockInferResult(*mock_infer_context.infer_data_.options_)};

    cb::OnCompleteFn& stream_callback{mock_infer_context.async_callback_func_};

    EXPECT_CALL(
        dynamic_cast<cb::MockClientBackend&>(
            *mock_infer_context.infer_backend_),
        AsyncStreamInfer(testing::_, testing::_, testing::_))
        .WillOnce(
            [&mock_infer_result, &stream_callback](
                const cb::InferOptions& options,
                const std::vector<cb::InferInput*>& inputs,
                const std::vector<const cb::InferRequestedOutput*>& outputs)
                -> cb::Error {
              stream_callback(mock_infer_result);
              return cb::Error::Success;
            });

    mock_infer_context.SendRequest(request_id, delayed, sequence_id);

    CHECK(mock_infer_context.thread_stat_->request_records_.size() == 1);
    CHECK(
        mock_infer_context.thread_stat_->request_records_[0].sequence_id_ ==
        sequence_id);
  }
}

}}  // namespace triton::perfanalyzer
