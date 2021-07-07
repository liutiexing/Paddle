// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gperftools/profiler.h>
#include <chrono>
#include <iostream>
#include <string>

#include <set>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

namespace paddle {
namespace framework {

class RuntimeContextV2 {
 public:
  RuntimeContextV2(std::vector<std::vector<Variable*>>& in_values,   // NOLINT
                   std::vector<std::vector<Variable*>>& out_values,  // NOLINT
                   const std::map<std::string, size_t>& in_name_map,
                   const std::map<std::string, size_t>& out_name_map)
      : input_values(std::move(in_values)),
        output_values(std::move(out_values)),
        input_name_map(in_name_map),
        output_name_map(out_name_map) {}
  std::vector<std::vector<Variable*>> input_values;
  std::vector<std::vector<Variable*>> output_values;
  const std::map<std::string, size_t>& input_name_map;
  const std::map<std::string, size_t>& output_name_map;
};

class ExecutionContextV2 : public ExecutionContext {
 public:
  ExecutionContextV2(const OperatorBase& op, const Scope& scope,
                     const platform::DeviceContext& device_context,
                     const RuntimeContextV2& ctx)
      : ExecutionContext(op, scope, device_context, RuntimeContext({}, {})),
        ctx_(ctx) {}

  const std::vector<Variable*> MultiInputVar(const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto it = ctx_.input_name_map.find(name);
    if (it == ctx_.input_name_map.end()) {
      return {};
    }
    // return {it->second.begin(), it->second.end()};
    return ctx_.input_values[it->second];
  }

  std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto it = ctx_.output_name_map.find(name);
    if (it == ctx_.output_name_map.end()) {
      return {};
    }
    // return it->second;
    return ctx_.output_values[it->second];
  }

  std::vector<std::string> InNameList() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(ctx_.output_name_map.size());

    for (auto& input : ctx_.output_name_map) {
      vec_temp.push_back(input.first);
    }

    return vec_temp;
  }

  const Variable* InputVar(const std::string& name) const {
    LogVarUsageIfUnusedVarCheckEnabled(name);

    auto it = ctx_.input_name_map.find(name);
    if (it == ctx_.input_name_map.end()) return nullptr;

    PADDLE_ENFORCE_LE(
        ctx_.input_values[it->second].size(), 1UL,
        platform::errors::InvalidArgument(
            "Operator %s's input %s should contain only one variable.",
            GetOp().Type(), name));
    return ctx_.input_values[it->second].empty()
               ? nullptr
               : ctx_.input_values[it->second][0];
  }

  Variable* OutputVar(const std::string& name) const {
    auto it = ctx_.output_name_map.find(name);
    if (it == ctx_.output_name_map.end()) return nullptr;

    PADDLE_ENFORCE_LE(
        ctx_.output_values[it->second].size(), 1UL,
        platform::errors::InvalidArgument(
            "Operator %s's output %s should contain only one variable.",
            GetOp().Type(), name));
    return ctx_.output_values[it->second].empty()
               ? nullptr
               : ctx_.output_values[it->second][0];
  }

  const RuntimeContextV2& ctx_;
};

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const RuntimeContextV2& ctx)
      : op_(op), ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.input_name_map;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = ctx_.input_values[it->second];
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(
        in.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input %s should not contain more than one inputs.", name));
    return in[0] != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    // has only one output
    const auto& outs = ctx_.output_name_map;
    auto it = outs.find(name);
    if (it == outs.end()) {
      return false;
    }
    const auto& out = ctx_.output_values[it->second];
    if (out.size() == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        out.size(), 1UL,
        platform::errors::InvalidArgument(
            "Output %s should not contain more than one outputs.", name));
    return out[0] != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    const auto& ins = ctx_.input_name_map;
    auto it = ins.find(name);
    if (it == ins.end() || ctx_.input_values[it->second].empty()) {
      return false;
    }
    for (auto& input : ctx_.input_values[it->second]) {
      if (input == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    const auto& outs = ctx_.output_name_map;
    auto it = outs.find(name);
    if (it == outs.end() || ctx_.output_values[it->second].empty()) {
      return false;
    }
    for (auto& output : ctx_.output_values[it->second]) {
      if (output == nullptr) {
        return false;
      }
    }
    return true;
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  std::vector<std::string> Inputs(const std::string& name) const override {
    return op_.Inputs(name);
  }

  std::vector<std::string> Outputs(const std::string& name) const override {
    return op_.Outputs(name);
  }

  std::string GetInputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(idx, op_proto->inputs().size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          op_.Type(), idx, op_proto->inputs().size()));
    return op_proto->inputs()[idx].name();
  }

  std::string GetOutputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(
        idx, op_proto->outputs().size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            op_.Type(), idx, op_proto->outputs().size()));
    return op_proto->outputs()[idx].name();
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.input_name_map.find(in);
    auto out_it = ctx_.output_name_map.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.input_name_map.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.output_name_map.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, ctx_.input_values[in_it->second].size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          ctx_.input_values[in_it->second].size(), i));
    PADDLE_ENFORCE_LT(j, ctx_.output_values[out_it->second].size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          ctx_.output_values[out_it->second].size(), j));

    Variable* in_var = ctx_.input_values[in_it->second][i];
    Variable* out_var = ctx_.output_values[out_it->second][j];

    PADDLE_ENFORCE_EQ(
        in_var->Type(), out_var->Type(),
        platform::errors::InvalidArgument(
            "The type of input (%s) and output (%s) are inconsistent.", in,
            out));

    if (in_var->IsType<framework::SelectedRows>()) {
      auto& in_sele_rows = in_var->Get<framework::SelectedRows>();
      auto out_sele_rows = out_var->GetMutable<framework::SelectedRows>();
      out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
      out_sele_rows->set_rows(in_sele_rows.rows());
      out_sele_rows->set_height(in_sele_rows.height());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto& in_lod_tensor = in_var->Get<framework::LoDTensor>();
      auto* out_lod_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_lod_tensor->Resize(in_lod_tensor.dims());
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows."));
    }
  }

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override {
    auto in_it = ctx_.input_name_map.find(in);
    auto out_it = ctx_.output_name_map.find(out);
    PADDLE_ENFORCE_NE(in_it, ctx_.input_name_map.end(),
                      platform::errors::NotFound(
                          "Input [%s] found error in Op [%s]", in, op_.Type()));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.output_name_map.end(),
        platform::errors::NotFound("Output [%s] found error in Op [%s]", out,
                                   op_.Type()));

    auto& in_var_list = ctx_.input_values[in_it->second];
    auto& out_var_list = ctx_.output_values[out_it->second];

    PADDLE_ENFORCE_EQ(
        in_var_list.size(), out_var_list.size(),
        platform::errors::PreconditionNotMet(
            "Op [%s]: Input var size should be equal with output var size",
            op_.Type()));

    auto& out_var_names = op_.Outputs(out);

    for (size_t i = 0; i < in_var_list.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) {
        continue;
      }

      Variable* in_var = in_var_list[i];
      if (!in_var->IsType<LoDTensor>()) return;
      Variable* out_var = out_var_list[i];
      PADDLE_ENFORCE_EQ(out_var->IsType<LoDTensor>(), true,
                        platform::errors::PreconditionNotMet(
                            "The %d-th output of Output(%s) must be LoDTensor.",
                            i, out_var_names[i]));
      auto& in_tensor = in_var->Get<LoDTensor>();
      auto* out_tensor = out_var->GetMutable<LoDTensor>();
      out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_MKLDNN
      if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
        out_tensor->set_layout(in_tensor.layout());
    }
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    auto in_it = ctx_.input_name_map.find(in);
    auto out_it = ctx_.output_name_map.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.input_name_map.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.output_name_map.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, ctx_.input_values[in_it->second].size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          ctx_.input_values[in_it->second].size(), i));
    PADDLE_ENFORCE_LT(j, ctx_.output_values[out_it->second].size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          ctx_.output_values[out_it->second].size(), j));

    Variable* in_var = ctx_.input_values[in_it->second].at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = ctx_.output_values[out_it->second].at(j);
    PADDLE_ENFORCE_EQ(
        out_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The %zu-th output of Output(%s) must be LoDTensor.", j, out));
    auto& in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out Tensor?
#ifdef PADDLE_WITH_MKLDNN
    // Fix me: ugly workaround below
    // Correct solution:
    //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
    //    layout of output tensor should be set "manually" in Compute()
    //    of each OPKernel. The reason layout should NOT be shared between
    //    input and output "automatically" (now by InferShape()->ShareLoD())
    //    is that layout transform may occur after InferShape().
    // Workaround:
    //    Skip set_layout() when input layout is kMKLDNN
    //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
    //    OPKernel. In all MKLDNN OPkernel, set_layout(kMKLDNN) should be called
    //    in Compute()
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  void SetLoDLevel(const std::string& out, int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  bool IsRuntime() const override { return true; }

  // TODO(paddle-dev): Can this be template?
  std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = InputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = OutputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  DDim GetInputDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(%s) should hold one element, but now it holds %zu elements.",
            name, vars.size()));
    return this->GetDim(vars[0]);
  }

  std::vector<DDim> GetInputsDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    return GetDims(vars);
  }

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override {
    return GetVarTypes(InputVars(name));
  }

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override {
    return GetVarTypes(OutputVars(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    // std::cerr << "set out dim" << std::endl;
    auto& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument("Output(%s) should hold one element, "
                                          "but now it holds %zu elements.",
                                          name, vars.size()));
    SetDim(vars[0], dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }

 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input variable is nullptr."));
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only LoDTensor or SelectedRows support 'GetDim', but input "
          "Variable's type is %s.",
          ToTypeName(var->Type())));
    }
  }

  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const {
    std::vector<DDim> ret;
    ret.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(ret),
                   [this](Variable* var) { return this->GetDim(var); });
    return ret;
  }

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetRepeatedDims method only ban be used in compile time."));
  }

  void SetDim(Variable* var, const DDim& dim) {
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Variable type error, expect LoDTensor or SelectedRows, but received "
          "(%s).",
          ToTypeName(var->Type())));
    }
  }

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size(),
                      platform::errors::InvalidArgument(
                          "The number of input variables do not match the "
                          "number of input dimensions, the number of variables "
                          "is %zu, the number of dimensions is %zu.",
                          length, dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetRepeatedDims method only can be used in compile time."));
  }

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(vars.size());
    std::transform(vars.begin(), vars.end(), retv.begin(),
                   std::bind(std::mem_fn(&RuntimeInferShapeContext::GetVarType),
                             this, std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(Variable* var) const {
    return ToVarType(var->Type());
  }

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const {
    auto it = ctx_.input_name_map.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.input_name_map.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the input (%s).", op_.Type(), name));
    return ctx_.input_values[it->second];
  }

  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.output_name_map.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.output_name_map.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
    return ctx_.output_values[it->second];
  }

  const OperatorBase& op_;
  const RuntimeContextV2& ctx_;
};

framework::ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    std::cout << "open file " << file_name << " faild!" << std::endl;
  }
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  ProgramDesc program_desc(buffer);
  return program_desc;
}

// ============================ ExecutorGraph =====================================
using Index = size_t;
using Num = size_t;
using StreamId = int;

enum class SchedType : int8_t {
  UNKNOWN = -1,
  CPU_COMPUTE = 0,
  ACCELERATOR_COMPUTE,
  DATA_COPY
};

struct ImmutableSchedInfo {
  SchedType sched_type{SchedType::UNKNOWN};
  StreamId stream_id{0};
  std::vector<std::vector<Index>> output_users;
};

using OpKernelFunc = std::function<void(const ExecutionContext&)>;

struct InstructionFunction {
  OpKernelFunc kernel_func = nullptr; // set on the first-run
  std::unique_ptr<OperatorBase> operator_ptr;
  
  InstructionFunction() = default;
 
  InstructionFunction(InstructionFunction&& inst) {
    kernel_func = std::move(inst.kernel_func);
    operator_ptr = std::move(inst.operator_ptr);
  }
};

struct ExecutionInfo {
  std::vector<Index> inputs;
  std::vector<Index> outputs;
  // input_values, output_values, input_indexes and output_indexes will be 
  // removed after Compute Unit Refactoring
  std::vector<std::vector<Index>> input_values;
  std::vector<std::vector<Index>> output_values;
  std::map<std::string, Index> input_indexes;
  std::map<std::string, Index> output_indexes;
  platform::Place place; // set by the placement-info, adjust on the first-run
  InstructionFunction function;
};

struct Instruction {
  ImmutableSchedInfo sched_info;
  ExecutionInfo exec_info;
};

struct VariableMetaInfo {
  std::string name;
  proto::VarType::Type type{proto::VarType::RAW};
  
  VariableMetaInfo(const std::string& n, proto::VarType::Type t) : name(n), type(t) {}
};

struct VersionedVariableInfo {
  Index metainfo;
  size_t version{1};

  VersionedVariableInfo(Index m, size_t v) : metainfo(m), version(v) {}
};

enum class VariableStatus : uint8_t {
  UNALLOCATED,
  ALLOCATED,
  ASSIGNED
};

struct InstructionState {
  std::atomic<Num> unallocated_input_num{std::numeric_limits<Num>::max()};
  std::atomic<Num> unassigned_input_num{std::numeric_limits<Num>::max()};

  InstructionState(Num unalloc, Num unassign) 
      : unallocated_input_num(unalloc),
        unassigned_input_num(unassign) {}

  InstructionState(const InstructionState& ) = delete;

  InstructionState(InstructionState&& state) 
      : unallocated_input_num(state.unallocated_input_num.load()),
        unassigned_input_num(state.unassigned_input_num.load()) {}
};

struct Register {
  std::unique_ptr<Variable> var{nullptr};
  VariableStatus var_status{VariableStatus::UNALLOCATED};
};

struct GraphBuildOptions {
  bool is_startup_prog = false;
};

class ExecutorGraph {
 public:
  ExecutorGraph(const BlockDesc& block, const GraphBuildOptions& options);

  std::map<std::string, proto::VarType::Type> GetParameterTypes();

 private:
  friend class FirstRunExecutorState;
  friend class ExecutorState;
  friend class AsyncExecutor;

  void BuildVariableInfo(const BlockDesc& block); 

  void BuildInstructionInfo(const BlockDesc& block); 

  void PopulateSchedInfo();

  void PopulateExecInfo(const BlockDesc& block, 
                        const std::vector<platform::Place>& placement_info); 

  void BuildTriggerInstructionInfo();
  
  std::unique_ptr<RuntimeContextV2> BuildRunCtx(const std::vector<Register>& registers, 
                                                Index inst) const;

  std::vector<Instruction> instructions_;
  std::vector<VariableMetaInfo> var_infos_;
  std::vector<VersionedVariableInfo> versioned_var_infos_;
  std::vector<Index> parameters_;
  std::vector<Index> inputs_;
  std::vector<Index> outputs_;
  std::unordered_map<std::string, Index> varname2idx_;
};

ExecutorGraph::ExecutorGraph(const BlockDesc& block, 
                             const GraphBuildOptions& options) {
  BuildVariableInfo(block);
  BuildInstructionInfo(block);
  if (options.is_startup_prog) {
    for (Index i = 0; i < var_infos_.size(); ++i) {
      parameters_.push_back(i);
    }
  }
}

std::map<std::string, proto::VarType::Type> ExecutorGraph::GetParameterTypes() {
  std::map<std::string, proto::VarType::Type> name_types;
  for (Index i : parameters_) {
    name_types[var_infos_[i].name] = var_infos_[i].type;
  }
  return name_types;
}

void ExecutorGraph::BuildInstructionInfo(const BlockDesc& block) { 
  size_t op_num = block.AllOps().size();
  std::vector<platform::Place> placement_info(op_num, platform::CUDAPlace(0));
  instructions_.resize(op_num + 1);
  PopulateExecInfo(block, placement_info);
  //PopulateSchedInfo();
}

void ExecutorGraph::PopulateExecInfo(const BlockDesc& block, 
                                     const std::vector<platform::Place>& placement_info) {
  const auto all_ops = block.AllOps();
  std::unordered_map<std::string, Index> name2version;
  for (size_t i = 0; i < all_ops.size(); ++i) {
    const OpDesc& opdesc = *(all_ops[i]);
    auto& exec_info = instructions_[i + 1].exec_info;
    std::set<Index> input_dedup;
    exec_info.input_values.reserve(opdesc.Inputs().size());
    for (const auto& var_name_map : opdesc.Inputs()) {
      exec_info.input_indexes[var_name_map.first] = exec_info.input_values.size();
      exec_info.input_values.push_back({});
      auto& idx_vec = exec_info.input_values.back();
      idx_vec.reserve(var_name_map.second.size());
      for (const auto& var_name : var_name_map.second) {
        const auto& iter = varname2idx_.find(var_name);
        Index idx = iter->second;
        idx_vec.push_back(idx);
        if (name2version.find(var_name) == name2version.end()) {
          versioned_var_infos_.emplace_back(idx, 1);
          name2version[var_name] = versioned_var_infos_.size() - 1;
        }
        input_dedup.insert(name2version[var_name]);
      }
    }
    exec_info.inputs.reserve(input_dedup.size());
    exec_info.inputs.insert(exec_info.inputs.begin(), input_dedup.begin(), input_dedup.end());
    std::set<Index> output_dedup;
    exec_info.output_values.reserve(opdesc.Outputs().size());
    for (const auto& var_name_map : opdesc.Outputs()) {
      exec_info.output_indexes[var_name_map.first] = exec_info.output_values.size();
      exec_info.output_values.push_back({});
      auto& idx_vec = exec_info.output_values.back();
      idx_vec.reserve(var_name_map.second.size());
      for (const auto& var_name : var_name_map.second) {
        const auto& iter = varname2idx_.find(var_name);
        Index idx = iter->second;
        idx_vec.push_back(idx);
        if (name2version.find(var_name) == name2version.end()) {
          versioned_var_infos_.emplace_back(idx, 1);
        }
        else {
          auto old_version = versioned_var_infos_[name2version[var_name]].version;
          versioned_var_infos_.emplace_back(idx, old_version + 1);
        }
        name2version[var_name] = versioned_var_infos_.size() - 1;
        output_dedup.insert(name2version[var_name]);
      }
    }
    exec_info.outputs.reserve(output_dedup.size());
    exec_info.outputs.insert(exec_info.outputs.begin(), output_dedup.begin(), output_dedup.end());
    exec_info.place = placement_info[i];
    exec_info.function.operator_ptr = OpRegistry::CreateOp(opdesc);
  }
}

void ExecutorGraph::PopulateSchedInfo() {
  std::vector<std::vector<Index>> var_users(versioned_var_infos_.size());
  for (Index i = 1; i < instructions_.size(); ++i) {
    for (Index input : instructions_[i].exec_info.inputs) {
      var_users[input].push_back(i);
    }
  }
  
  for (Index i = 1; i < instructions_.size(); ++i) {
    auto& output_users = instructions_[i].sched_info.output_users;
    output_users.reserve(instructions_[i].exec_info.outputs.size());
    for (Index output : instructions_[i].exec_info.outputs) {
      output_users.push_back(var_users[output]);
    }
  }
}

void ExecutorGraph::BuildVariableInfo(const BlockDesc& block) {
  var_infos_.reserve(block.AllVars().size());
  for (const auto& var : block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    varname2idx_[var->Name()] = var_infos_.size();
    var_infos_.emplace_back(var->Name(), var->GetType());
  }
}

void ExecutorGraph::BuildTriggerInstructionInfo() {
  std::set<Index> out_vars;
  out_vars.insert(parameters_.begin(), parameters_.end());
  out_vars.insert(inputs_.begin(), inputs_.end());
  std::unordered_map<Index, Index> out_ver_vars;
  for (Index i = 0, cur = 1; i < versioned_var_infos_.size(); ++i) {
    if (versioned_var_infos_[i].version == 1 &&
        out_vars.find(versioned_var_infos_[i].metainfo) != out_vars.end()) {
      out_ver_vars[i] = cur++;
    }
  }
  auto& var_users = instructions_[0].sched_info.output_users;
  var_users.resize(out_vars.size() + 1);
  for (Index i = 1; i < instructions_.size(); ++i) {
    if (instructions_[i].exec_info.inputs.size() == 0) {
      var_users[0].push_back(i);
      continue;
    }
    for (Index input : instructions_[i].exec_info.inputs) {
      const auto iter = out_ver_vars.find(input);
      if (iter != out_ver_vars.end()) {
        var_users[iter->second].push_back(i);
      }
    }
  }
}

std::unique_ptr<RuntimeContextV2> ExecutorGraph::BuildRunCtx(
        const std::vector<Register>& registers, Index inst_idx) const {
  const auto& exec_info = instructions_[inst_idx].exec_info;
  std::vector<std::vector<Variable*>> input_values;
  std::vector<std::vector<Variable*>> output_values;  
 
  input_values.resize(exec_info.input_indexes.size());
  for (const auto& input : exec_info.input_indexes) {
    auto& vars = input_values[input.second];
    for (Index var_idx : exec_info.input_values[input.second]) {
      vars.push_back(registers[var_idx].var.get());
    }
  }
  output_values.resize(exec_info.output_indexes.size());
  for (const auto& output : exec_info.output_indexes) {
    auto& vars = output_values[output.second];
    for (Index var_idx : exec_info.output_values[output.second]) {
      vars.push_back(registers[var_idx].var.get());
    }
  }
  std::unique_ptr<RuntimeContextV2> run_ctx(new RuntimeContextV2(input_values, 
                                                                 output_values, 
                                                                 exec_info.input_indexes,
                                                                 exec_info.output_indexes));
  return std::move(run_ctx);
}

// ============================ ExecutorState =====================================
class ExecutorState {
 public:
  ExecutorState(const ExecutorGraph& graph, 
                const std::vector<Variable*>& params);

  void Run(const std::vector<Variable*>& inputs, 
           std::vector<Variable*>& outputs);

  std::vector<std::string> GetInputNames();

  std::vector<std::string> GetOutputNames();

  std::vector<std::string> GetParameterNames();

 private:
  void ProcessTriggerInstruction(std::vector<Index>& ready_insts);

  void ProcessReadyInstructions(std::vector<Index>& ready_insts);

  std::vector<InstructionState> inst_states_;
  std::vector<Register> registers_;
  const ExecutorGraph& graph_;
};

ExecutorState::ExecutorState(const ExecutorGraph& graph, 
                             const std::vector<Variable*>& params)
    : graph_(graph) {
  // registers
  registers_.resize(graph_.var_infos_.size());
  for (Index i = 0; i < registers_.size(); ++i) {
    registers_[i].var = std::unique_ptr<Variable>(new Variable);
    InitializeVariable(registers_[i].var.get(), graph_.var_infos_[i].type);
  }
  // inst state
  for (Index i = 0; i < graph_.instructions_.size(); ++i) {
    const auto& inst = graph_.instructions_[i];
    inst_states_.emplace_back(inst.exec_info.inputs.size(), 
                              inst.exec_info.inputs.size());
  }
  // copy parameters
  for (Index i = 0; i < params.size(); ++i) {
    const auto& src_tensor = params[i]->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto param_idx = graph_.parameters_[i];
    auto* dest_tensor = registers_[param_idx].var->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
    registers_[param_idx].var_status = VariableStatus::ASSIGNED;
  }
}

void ExecutorState::Run(const std::vector<Variable*>& inputs,
                        std::vector<Variable*>& outputs) {
  // copy inputs
  for (Index i = 0; i < inputs.size(); ++i) {
    const auto& src_tensor = inputs[i]->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto* dest_tensor = registers_[graph_.inputs_[i]].var->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
    registers_[graph_.inputs_[i]].var_status = VariableStatus::ASSIGNED;
  }
  // run insts
  std::vector<Index> ready_insts;
  ProcessTriggerInstruction(ready_insts);
  ProcessReadyInstructions(ready_insts);
  // copy outputs
  for (Index i = 0; i < outputs.size(); ++i) {
    const auto& src_tensor = registers_[graph_.outputs_[i]].var->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto* dest_tensor = outputs[i]->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
  }
}

void ExecutorState::ProcessTriggerInstruction(std::vector<Index>& ready_insts) {
  const auto& sched_info = graph_.instructions_[0].sched_info;
  for (auto user : sched_info.output_users[0]) {
    ready_insts.push_back(user);
  }
  for (Index i = 1; i < sched_info.output_users.size(); ++i) {
    for (auto user : sched_info.output_users[i]) {
      if (inst_states_[user].unassigned_input_num.fetch_sub(1) == 1) {
        ready_insts.push_back(user);
      }
    }
  }
}

void ExecutorState::ProcessReadyInstructions(std::vector<Index>& ready_insts) {
  auto& dev_ctx_pool = platform::DeviceContextPool::Instance();
  Scope null_scope;
  while (!ready_insts.empty()) {
    std::vector<Index> cur_ready_insts;
    cur_ready_insts.swap(ready_insts);
    for (auto inst : cur_ready_insts) {
      // exec ready inst
      const auto& exec_info = graph_.instructions_[inst].exec_info;
      const auto& op_base = *(exec_info.function.operator_ptr);
      auto run_ctx_ptr = graph_.BuildRunCtx(registers_, inst);
      RuntimeInferShapeContext infershape_ctx(op_base, *run_ctx_ptr);
      const auto& op_with_kernel = 
          static_cast<const framework::OperatorWithKernel&>(op_base);
      op_with_kernel.InferShape(&infershape_ctx);
      const auto& dev_ctx = *(dev_ctx_pool.Get(exec_info.place));
      std::unique_ptr<ExecutionContextV2> exec_ctx_ptr(
          new ExecutionContextV2(op_base, null_scope, dev_ctx, *run_ctx_ptr));
      exec_info.function.kernel_func(*exec_ctx_ptr);
      // sched ready inst
      const auto& sched_info = graph_.instructions_[inst].sched_info;
      for (Index i = 0; i < sched_info.output_users.size(); ++i) {
        for (auto user : sched_info.output_users[i]) {
          if (inst_states_[user].unassigned_input_num.fetch_sub(1) == 1) {
            ready_insts.push_back(user);
          }
        }
      }
    }
  }
}

class FirstRunExecutorState {
 public:
  FirstRunExecutorState(const std::vector<std::string>& param_names, 
                        const std::vector<Variable*>& params,
                        ExecutorGraph& graph);

  void Run(const std::vector<std::string>& input_names, 
           const std::vector<Variable*>& inputs, 
           const std::vector<std::string>& output_names, 
           std::vector<Variable*>& outputs);

 private:
  std::vector<Register> registers_;
  ExecutorGraph& graph_;
};

FirstRunExecutorState::FirstRunExecutorState(
    const std::vector<std::string>& param_names, 
    const std::vector<Variable*>& params, 
    ExecutorGraph& graph) : graph_(graph) {
  // set graph parameters
  graph_.parameters_.clear();
  graph_.parameters_.resize(param_names.size());
  for (Index i = 0; i < param_names.size(); ++i) {
    auto iter = graph_.varname2idx_.find(param_names[i]);
    graph_.parameters_[i] = iter->second;
  }
  // registers
  registers_.resize(graph_.var_infos_.size());
  for (Index i = 0; i < registers_.size(); ++i) {
    registers_[i].var = std::unique_ptr<Variable>(new Variable);
    InitializeVariable(registers_[i].var.get(), graph_.var_infos_[i].type);
  }
  // copy parameters
  for (Index i = 0; i < params.size(); ++i) {
    const auto& src_tensor = params[i]->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto param_idx = graph_.varname2idx_[param_names[i]];
    auto* dest_tensor = registers_[param_idx].var->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
    registers_[param_idx].var_status = VariableStatus::ASSIGNED;
  }
}

int convert(const platform::Place& place);

void FirstRunExecutorState::Run(const std::vector<std::string>& input_names, 
                                const std::vector<Variable*>& inputs, 
                                const std::vector<std::string>& output_names, 
                                std::vector<Variable*>& outputs) {
  // set graph inputs/outputs
  graph_.inputs_.resize(input_names.size());
  for (Index i = 0; i < input_names.size(); ++i) {
    auto iter = graph_.varname2idx_.find(input_names[i]);
    graph_.inputs_[i] = iter->second;
  }
  graph_.outputs_.resize(output_names.size());
  for (Index i = 0; i < output_names.size(); ++i) {
    auto iter = graph_.varname2idx_.find(output_names[i]);
    graph_.outputs_[i] = iter->second;
  }
 
  // copy inputs
  for (Index i = 0; i < inputs.size(); ++i) {
    const auto& src_tensor = inputs[i]->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto* dest_tensor = registers_[graph_.inputs_[i]].var->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
    registers_[graph_.inputs_[i]].var_status = VariableStatus::ASSIGNED;
  }
  // build and run
  auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
  auto& dev_ctx_pool = platform::DeviceContextPool::Instance();
  Scope null_scope;
  size_t inst_num = graph_.instructions_.size();
  std::vector<std::map<platform::Place, Index>> copy_vars;
  copy_vars.resize(graph_.versioned_var_infos_.size());
  for (Index inst = 1; inst < inst_num; ++inst) {
    auto& exec_info = graph_.instructions_[inst].exec_info;
    const auto& op_base = *(exec_info.function.operator_ptr);
    auto kernels_iter = all_op_kernels.find(op_base.Type());
    auto run_ctx_ptr = graph_.BuildRunCtx(registers_, inst);
    RuntimeInferShapeContext infershape_ctx(op_base, *run_ctx_ptr);
    const auto& op_with_kernel = 
        static_cast<const framework::OperatorWithKernel&>(op_base);
    op_with_kernel.InferShape(&infershape_ctx);
    const auto& dev_ctx = *(dev_ctx_pool.Get(exec_info.place));
    std::unique_ptr<ExecutionContextV2> exec_ctx_ptr(
        new ExecutionContextV2(op_base, null_scope, dev_ctx, *run_ctx_ptr));
    OpKernelType expected_kernel_type = op_with_kernel.GetExpectedKernelType(*exec_ctx_ptr);
    OperatorWithKernel::OpKernelMap& kernels = kernels_iter->second;
    auto kernel_iter = kernels.find(expected_kernel_type);
    if (kernel_iter == kernels.end()) {
      std::cout << "no op kernel " << std::endl;
    }
    exec_info.function.kernel_func = OpKernelFunc(kernel_iter->second);
    // check inputs
    bool inputs_changed = false;
    for (auto& orig_input : graph_.instructions_[inst].exec_info.inputs) {
      Index orig_var_idx = graph_.versioned_var_infos_[orig_input].metainfo;
      const Tensor& input_tensor = registers_[orig_var_idx].var->Get<LoDTensor>();
      if (!input_tensor.IsInitialized()) {
        continue;
      }
      const auto orig_var_name = graph_.var_infos_[orig_var_idx].name;
      auto var_type = op_with_kernel.GetKernelTypeForVar(orig_var_name, 
                                                         input_tensor, 
                                                         expected_kernel_type);
      if (platform::is_same_place(var_type.place_, expected_kernel_type.place_)) {
        continue;
      }
      inputs_changed = true;
      Index copy_var_idx = 0;
      Index copy_ver_var_idx = 0;
      auto copy_var_place = expected_kernel_type.place_;
      auto& copy_vars_for_orig = copy_vars[orig_input];
      auto iter = copy_vars_for_orig.find(copy_var_place);
      if (iter != copy_vars_for_orig.end()) {
        copy_ver_var_idx = iter->second;
        copy_var_idx = graph_.versioned_var_infos_[copy_ver_var_idx].metainfo;
      }
      else {
        // add data-copy var
        std::string copy_var_name = "copy_" + orig_var_name;
        copy_var_idx = graph_.var_infos_.size();
        graph_.var_infos_.emplace_back(copy_var_name, 
                                       graph_.var_infos_[orig_var_idx].type);
        copy_ver_var_idx = graph_.versioned_var_infos_.size();
        copy_vars_for_orig[copy_var_place] = copy_ver_var_idx;
        graph_.versioned_var_infos_.emplace_back(copy_var_idx, 1);
        registers_.push_back(Register());
        auto v = new Variable();
        v->GetMutable<LoDTensor>();
        registers_.back().var.reset(v);
        // add data-copy op
        Index copy_inst_idx = graph_.instructions_.size();
        graph_.instructions_.push_back(Instruction());
        auto& copy_inst = graph_.instructions_.back();
        copy_inst.exec_info.inputs.push_back(orig_input);
        copy_inst.exec_info.outputs.push_back(copy_ver_var_idx);
        copy_inst.exec_info.input_indexes["X"] = 0;
        copy_inst.exec_info.input_values.push_back({orig_var_idx});
        copy_inst.exec_info.output_indexes["Out"] = 0;
        copy_inst.exec_info.output_values.push_back({copy_var_idx});
        copy_inst.exec_info.place = graph_.instructions_[inst].exec_info.place;
        // exec data-copy op
        VariableNameMap copy_input_map;
        VariableNameMap copy_output_map;
        AttributeMap copy_attr_map;
        copy_input_map["X"] = {orig_var_name};
        copy_output_map["Out"] = {copy_var_name};
        copy_attr_map["dst_place_type"] = convert(graph_.instructions_[inst].exec_info.place);
        auto& copy_op_info = OpInfoMap::Instance().Get("memcpy");
        auto copy_op_base = copy_op_info.Creator()("memcpy", copy_input_map, 
                                                   copy_output_map, copy_attr_map);
        auto copy_run_ctx_ptr = graph_.BuildRunCtx(registers_, copy_inst_idx);
        RuntimeInferShapeContext copy_infershape_ctx(*copy_op_base, *copy_run_ctx_ptr);
        const auto& copy_op_with_kernel = static_cast<const OperatorWithKernel&>(*copy_op_base);
        copy_op_with_kernel.InferShape(&copy_infershape_ctx);
        const auto& copy_dev_ctx = *(dev_ctx_pool.Get(copy_inst.exec_info.place));
        ExecutionContextV2 copy_exec_ctx(*copy_op_base, null_scope, copy_dev_ctx, *copy_run_ctx_ptr);
        OpKernelType copy_expected_kernel = copy_op_with_kernel.GetExpectedKernelType(copy_exec_ctx);
        auto copy_kernels_iter = all_op_kernels.find("memcpy");
        OperatorWithKernel::OpKernelMap& copy_kernels = copy_kernels_iter->second;
        auto copy_kernel_iter = copy_kernels.find(copy_expected_kernel);
        if (copy_kernel_iter == copy_kernels.end()) {
          std::cout << "no copy kernel " << std::endl;
        }
        copy_inst.exec_info.function.operator_ptr.reset(copy_op_base);
        copy_inst.exec_info.function.kernel_func = OpKernelFunc(copy_kernel_iter->second);
        copy_inst.exec_info.function.kernel_func(copy_exec_ctx);
      }
      // adjust the input
      orig_input = copy_ver_var_idx;
      for (auto& indexes : graph_.instructions_[inst].exec_info.input_values) {
        for (auto& var_idx: indexes) {
          if (var_idx == orig_var_idx) {
            var_idx = copy_var_idx;
          }
        }
      }
    }
    // run kernel_func
    if (inputs_changed) {
      run_ctx_ptr = graph_.BuildRunCtx(registers_, inst);
      exec_ctx_ptr.reset(new ExecutionContextV2(op_base, null_scope, dev_ctx, *run_ctx_ptr));
    }
    graph_.instructions_[inst].exec_info.function.kernel_func(*exec_ctx_ptr);
    graph_.instructions_[inst].exec_info.place = expected_kernel_type.place_; // adjust place
  }
  graph_.BuildTriggerInstructionInfo();
  graph_.PopulateSchedInfo();
  // copy outputs
  for (Index i = 0; i < output_names.size(); ++i) {
    const auto& src_tensor = registers_[graph_.outputs_[i]].var->Get<framework::LoDTensor>();
    if (src_tensor.IsInitialized() == false) {
      continue;
    }
    auto* dest_tensor = outputs[i]->GetMutable<framework::LoDTensor>();
    dest_tensor->ShareDataWith(src_tensor);
  }
}

// ============================ AsyncExecutor =====================================
class AsyncExecutor {
 public:
  AsyncExecutor(const ProgramDesc& startup_prog, const ProgramDesc& main_prog) {
    InitParameters(startup_prog);
    GraphBuildOptions options;
    graph_.reset(new ExecutorGraph(main_prog.Block(0), options));
  }
  
  void TestProgramToGraph(const ProgramDesc& programdesc) {
    std::cout << "begin to translate ProgramDesc to ExecutorGraph " << std::endl;
    GraphBuildOptions options;
    ExecutorGraph graph(programdesc.Block(0), options);
    std::cout << "finish to translate ProgramDesc to ExecutorGraph " << std::endl;
  }

  void Run(const std::vector<std::string>& input_names,
           const std::vector<framework::Tensor>& inputs,
           const std::vector<std::string>& output_names,
           std::vector<framework::Tensor>& outputs);

  std::map<std::string, const Variable*> GetParameters();

 private:
  void InitParameters(const ProgramDesc& startup_prog); 

  std::vector<Variable*> GetParameterRefs();

  bool first_run_ = true;
  std::vector<std::string> parameter_names_;
  std::vector<std::unique_ptr<Variable>> parameters_;
  std::unique_ptr<ExecutorGraph> graph_;
};

void AsyncExecutor::InitParameters(const ProgramDesc& startup_prog) {
  GraphBuildOptions options;
  options.is_startup_prog = true;
  ExecutorGraph startup_graph(startup_prog.Block(0), options);

  auto param_types = startup_graph.GetParameterTypes();
  parameters_.reserve(param_types.size());
  parameter_names_.reserve(param_types.size());
  for (const auto& param_name_type : param_types) { 
    auto param = new Variable;
    InitializeVariable(param, param_name_type.second);
    parameters_.push_back(std::unique_ptr<Variable>(param));
    parameter_names_.push_back(param_name_type.first);
  }
  std::vector<Variable*> params(parameters_.size());
  for (Index i = 0; i < parameters_.size(); ++i) {
    params[i] = parameters_[i].get();
  }
  FirstRunExecutorState executor_state(parameter_names_, params, startup_graph);
  executor_state.Run({}, std::vector<Variable*>(), parameter_names_, params);
}

void AsyncExecutor::Run(const std::vector<std::string>& input_names,
                        const std::vector<framework::Tensor>& input_tensors,
                        const std::vector<std::string>& output_names,
                        std::vector<framework::Tensor>& output_tensors) {
  // prepare param, inputs, outputs
  std::vector<Variable*> params(parameters_.size());
  for (Index i = 0; i < parameters_.size(); ++i) {
    params[i] = parameters_[i].get();
  }
  std::vector<Variable*> inputs(input_names.size());
  std::vector<Variable> input_vars(input_names.size());
  for (Index i = 0; i < inputs.size(); ++i) {
    auto* dest_tensor = input_vars[i].GetMutable<LoDTensor>();
    dest_tensor->ShareDataWith(input_tensors[i]);
    inputs[i] = &input_vars[i];
  }
  std::vector<Variable*> outputs(output_names.size());
  std::vector<Variable> output_vars(output_names.size());
  for (Index i = 0; i < outputs.size(); ++i) {
    output_vars[i].GetMutable<LoDTensor>();
    outputs[i] = &output_vars[i];
  }

  if (first_run_) {
    FirstRunExecutorState exec_state(parameter_names_, params, *graph_.get());
    exec_state.Run(input_names, inputs, output_names, outputs);
    first_run_ = false;
  }
  else {
    ExecutorState exec_state(*graph_.get(), params);
    exec_state.Run(inputs, outputs);
  }
  // copy outputs
  for (Index i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = output_vars[i].Get<LoDTensor>();
    if (!platform::is_cpu_place(output_tensor.place())) {
        auto& pool = platform::DeviceContextPool::Instance();
        auto* dev_ctx = pool.Get(output_tensor.place());
        dev_ctx->Wait();
        TensorCopySync(output_tensor, platform::CPUPlace(), &output_tensors[i]);
        dev_ctx->Wait();
    }
    else {
      output_tensors[i].ShareDataWith(output_tensor);
    }
  }
}

std::vector<Variable*> AsyncExecutor::GetParameterRefs() {
  std::vector<Variable*> refs;
  refs.reserve(parameters_.size());
  for (auto& param : parameters_) {
    refs.push_back(param.get());
  }
  return refs;
}

// =========================================== =====================================

struct VariableScope {
  std::vector<std::unique_ptr<Variable>> var_list;
  std::map<std::string, size_t> name2id;
};

struct OpFuncNode {
  // int unsed;
  // std::map< std::string, std::vector<int> > input_index;
  // std::map< std::string, std::vector<int> > output_index;
  std::vector<std::vector<size_t>> input_index;
  std::vector<std::vector<size_t>> output_index;
  std::map<std::string, size_t> input_name_map;
  std::map<std::string, size_t> output_name_map;

  using OpKernelFunc = std::function<void(const ExecutionContext&)>;
  OpKernelFunc kernel_func_;
};

int convert(const platform::Place& place) {
  if (is_cpu_place(place)) {
    return 0;
  }
  if (is_gpu_place(place)) {
    return 1;
  }

  return -1;
}

void build_variable_scope(const framework::ProgramDesc& pdesc,
                          VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    // std::cerr << "var name "  << var->Name() << std::endl;

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
    }

    auto v = new Variable();
    // v->GetMutable<LoDTensor>();
    InitializeVariable(v, var->GetType());
    var_scope->var_list.push_back(std::unique_ptr<Variable>(v));
  }
}

void build_op_func_list(const framework::ProgramDesc& pdesc,
                        std::vector<OperatorBase*>& op_list,     // NOLINT
                        std::vector<OpFuncNode>& vec_func_list,  // NOLINT
                        VariableScope* var_scope,
                        const platform::Place& place) {
  auto& global_block = pdesc.Block(0);

  int inst_num = 0;
  for (auto& op : global_block.AllOps()) {
    // std::cerr << op->Type() << std::endl;
    // bool debug = op->Type() == "softmax_with_cross_entropy_grad";
    bool debug = false;

    // std::cerr << "create op" << std::endl;
    // auto op_base_u = OpRegistry::CreateOp(*op);
    auto& info = OpInfoMap::Instance().Get(op->Type());

    VariableNameMap inputs_1 = op->Inputs();
    VariableNameMap outputs_1 = op->Outputs();
    AttributeMap attrs_1 = op->GetAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&attrs_1);
    }
    auto op_base = info.Creator()(op->Type(), inputs_1, outputs_1, attrs_1);

    auto input_names = op->Inputs();
    auto output_names = op->Outputs();

    OpFuncNode op_func_node;

    // VariableValueMap ins_map;
    // std::map<std::string, std::vector<int> > ins_name2id;
    std::vector<std::vector<Variable*>> ins_value;
    std::vector<std::vector<size_t>> ins_index;
    std::map<std::string, size_t> ins_name_map;
    for (auto& var_name_item : input_names) {
      std::vector<Variable*> input_vars;
      std::vector<size_t> vec_ids;
      input_vars.reserve(var_name_item.second.size());
      for (auto& var_name : var_name_item.second) {
        auto it = var_scope->name2id.find(var_name);
        assert(it != var_scope->name2id.end());
        input_vars.push_back(var_scope->var_list[it->second].get());
        vec_ids.push_back(it->second);
      }
      ins_value.emplace_back(std::move(input_vars));
      ins_index.emplace_back(std::move(vec_ids));
      ins_name_map[var_name_item.first] = ins_index.size() - 1;
      // ins_map[ var_name_item.first ] = input_vars;
      // ins_name2id[ var_name_item.first ] = vec_ids;
    }
    if (debug) std::cerr << "1" << std::endl;

    // VariableValueMap outs_map;
    // std::map<std::string, std::vector<int> > outs_name2id;
    std::vector<std::vector<Variable*>> outs_value;
    std::vector<std::vector<size_t>> outs_index;
    std::map<std::string, size_t> outs_name_map;
    for (auto& var_name_item : output_names) {
      std::vector<Variable*> output_vars;
      std::vector<size_t> vec_ids;
      output_vars.reserve(var_name_item.second.size());
      for (auto& var_name : var_name_item.second) {
        auto it = var_scope->name2id.find(var_name);
        assert(it != var_scope->name2id.end());
        // std::cerr << it->second << "\t" << var_scope.var_list.size() <<
        // std::endl;
        output_vars.push_back(var_scope->var_list[it->second].get());
        vec_ids.push_back(it->second);
      }
      outs_value.emplace_back(std::move(output_vars));
      outs_index.emplace_back(std::move(vec_ids));
      outs_name_map[var_name_item.first] = outs_index.size() - 1;
      // outs_map[ var_name_item.first ] = output_vars;
      // //std::cerr << ToTypeName(output_vars[0]->Type() ) << std::endl;
      // outs_name2id[ var_name_item.first ] = vec_ids;
    }

    // op_func_node.input_index = ins_name2id;
    // op_func_node.output_index = outs_name2id;
    op_func_node.input_index = ins_index;
    op_func_node.input_name_map = ins_name_map;
    op_func_node.output_index = outs_index;
    op_func_node.output_name_map = outs_name_map;
    RuntimeContextV2 runtime_context(ins_value, outs_value, ins_name_map,
                                     outs_name_map);
    // runtime_context.inputs.swap( ins_map );
    // runtime_context.outputs.swap(  outs_map );
    // runtime_context.input_values.swap(ins_value);
    // runtime_context.input_name_map = ins_name_map;
    // runtime_context.output_values.swap(outs_value);
    // runtime_context.output_name_map = outs_name_map;
    // std::cerr << "create runtime context" << std::endl;
    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);
    // std::cerr << "fin infer shape" << std::endl;
    auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op->Type());
    PADDLE_ENFORCE_NE(
        kernels_iter, all_op_kernels.end(),
        platform::errors::Unavailable(
            "There are no kernels which are registered in the %s operator.",
            op->Type()));

    // std::cerr << "create kernel" << std::endl;
    using OpKernelFunc = std::function<void(const ExecutionContext&)>;
    using OpKernelMap =
        std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;
    if (debug) std::cerr << "2" << std::endl;
    OpKernelMap& kernels = kernels_iter->second;
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);
    Scope scope;
    auto exec_ctx =
        ExecutionContextV2(*op_base, scope, *dev_ctx, runtime_context);
    if (debug) std::cerr << "21" << std::endl;
    auto expected_kernel_key =
        dynamic_cast<const framework::OperatorWithKernel*>(op_base)
            ->GetExpectedKernelType(exec_ctx);
    if (debug) std::cerr << "22" << std::endl;
    // std::cerr << "22" << std::endl;

    // add transfer log
    // std::cerr << "in map size " << ins_map.size() << std::endl;
    // VariableValueMap&  ins_map_temp = runtime_context.inputs;
    auto ins_map_temp = runtime_context.input_name_map;
    // std::cerr << "ins map siz" << ins_map_temp.size() << std::endl;
    for (auto& var_name_item : ins_map_temp) {
      // std::cerr << "in name " << var_name_item.first << std::endl;
      // auto& vec_ids = ins_name2id[ var_name_item.first ];
      for (size_t i = 0;
           i < runtime_context.input_values[var_name_item.second].size(); ++i) {
        auto var = runtime_context.input_values[var_name_item.second][i];
        auto tensor_in = static_cast<const Tensor*>(&(var->Get<LoDTensor>()));
        if (!tensor_in->IsInitialized()) {
          continue;
        }
        // std::cerr << "i " << i << "\t" << tensor_in->IsInitialized() <<
        // std::endl;
        auto kernel_type_for_var =
            static_cast<const framework::OperatorWithKernel*>(op_base)
                ->GetKernelTypeForVar(var_name_item.first, *tensor_in,
                                      expected_kernel_key);
        if (debug) {
          std::cerr << "var name " << var_name_item.first << std::endl;
          std::cerr << expected_kernel_key.place_ << "\t"
                    << kernel_type_for_var.place_ << std::endl;
        }
        if (!platform::is_same_place(kernel_type_for_var.place_,
                                     expected_kernel_key.place_)) {
          if (debug) std::cerr << "add data transfer" << std::endl;
          // need trans place
          // add var in scope
          // add copy op
          std::string new_var_name =
              "temp_1" + std::to_string(var_scope->var_list.size() + 1);
          auto v = new Variable();
          v->GetMutable<LoDTensor>();
          var_scope->name2id[new_var_name] = var_scope->var_list.size();
          var_scope->var_list.push_back(std::unique_ptr<Variable>(v));

          VariableNameMap copy_in_map;
          // std::cerr << "ints name is " << input_names[var_name_item.first][i]
          //     << std::endl;
          copy_in_map["X"] = {input_names[var_name_item.first][i]};
          VariableNameMap copy_out_map;
          copy_out_map["Out"] = {new_var_name};
          AttributeMap attr_map;
          attr_map["dst_place_type"] = convert(place);

          // std::map< std::string, std::vector<int> > copy_ins_name2id;
          // copy_ins_name2id["X"] = ins_name2id[ var_name_item.first ];
          // std::map< std::string, std::vector<int> > copy_out_name2id;
          // copy_out_name2id["Out"] = { var_scope->name2id[new_var_name]};

          // vec_ids[i] = var_scope->name2id[new_var_name];
          // update out runtime_context
          op_func_node
              .input_index[op_func_node.input_name_map[var_name_item.first]]
                          [i] = var_scope->name2id[new_var_name];

          // VariableValueMap copy_ins_value_map;
          // copy_ins_value_map["X"] = { var };
          // VariableValueMap copy_outs_value_map;
          // copy_outs_value_map["Out"] = { v };

          auto& copy_info = OpInfoMap::Instance().Get("memcpy");
          auto copy_op = copy_info.Creator()("memcpy", copy_in_map,
                                             copy_out_map, attr_map);
          OpFuncNode copy_op_func_node;
          // copy_op_func_node.input_index = copy_ins_name2id;
          // copy_op_func_node.output_index = copy_out_name2id;
          copy_op_func_node.input_index.push_back(
              ins_index[ins_name_map[var_name_item.first]]);
          copy_op_func_node.input_name_map["X"] = 0;
          copy_op_func_node.output_index.push_back(
              {var_scope->name2id[new_var_name]});
          copy_op_func_node.output_name_map["Out"] = 0;
          std::vector<std::vector<Variable*>> in_values;
          std::vector<std::vector<Variable*>> out_values;
          in_values.push_back({var});
          out_values.push_back({v});
          RuntimeContextV2 copy_runtime_context(
              in_values, out_values, copy_op_func_node.input_name_map,
              copy_op_func_node.output_name_map);
          // copy_runtime_context.input_values.push_back({var});
          // copy_runtime_context.input_name_map["X"] = 0;
          // copy_runtime_context.output_values.push_back({v});
          // copy_runtime_context.output_name_map["Out"] = 0;
          // copy_runtime_context.inputs.swap( copy_ins_value_map );
          // copy_runtime_context.outputs.swap(  copy_outs_value_map );
          // std::cerr << "create runtime context" << std::endl;
          RuntimeInferShapeContext copy_infer_shape_ctx(*copy_op,
                                                        copy_runtime_context);
          if (debug) std::cerr << "before infer shape" << std::endl;
          static_cast<const framework::OperatorWithKernel*>(copy_op)
              ->InferShape(&copy_infer_shape_ctx);
          if (debug) std::cerr << "infer shape" << std::endl;
          // std::cerr << "fin infer shape" << std::endl;
          auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
          auto kernels_iter = all_op_kernels.find("memcpy");
          PADDLE_ENFORCE_NE(kernels_iter, all_op_kernels.end(),
                            platform::errors::Unavailable(
                                "There are no kernels which are registered in "
                                "the memcpy operator."));

          // std::cerr << "create kernel" << std::endl;
          using OpKernelFunc = std::function<void(const ExecutionContext&)>;
          using OpKernelMap = std::unordered_map<OpKernelType, OpKernelFunc,
                                                 OpKernelType::Hash>;

          OpKernelMap& kernels = kernels_iter->second;
          // auto place = platform::CPUPlace();
          // auto place = platform::CUDAPlace(0);

          platform::DeviceContextPool& pool =
              platform::DeviceContextPool::Instance();
          auto* dev_ctx = pool.Get(place);
          Scope scope;
          auto copy_exec_ctx = ExecutionContextV2(*copy_op, scope, *dev_ctx,
                                                  copy_runtime_context);
          if (debug) std::cerr << "21" << std::endl;
          auto expected_kernel_key =
              dynamic_cast<const framework::OperatorWithKernel*>(copy_op)
                  ->GetExpectedKernelType(copy_exec_ctx);
          if (debug) std::cerr << "22" << std::endl;
          // std::cerr << "22" << std::endl;
          auto kernel_iter = kernels.find(expected_kernel_key);
          copy_op_func_node.kernel_func_ = OpKernelFunc(kernel_iter->second);
          copy_op_func_node.kernel_func_(copy_exec_ctx);
          if (debug) std::cerr << "run exe ctx" << std::endl;

          op_list.push_back(copy_op);
          vec_func_list.push_back(copy_op_func_node);

          runtime_context.input_values[var_name_item.second][i] = v;
        }
      }
    }

    op_list.push_back(op_base);

    auto kernel_iter = kernels.find(expected_kernel_key);

    if (debug) std::cerr << "3" << std::endl;
    op_func_node.kernel_func_ = OpKernelFunc(kernel_iter->second);
    if (debug) std::cerr << "3-1" << std::endl;
    op_func_node.kernel_func_(exec_ctx);
    vec_func_list.push_back(op_func_node);
    if (debug) std::cerr << "5" << std::endl;
    ++inst_num;
  }
}

void exec_op_func_list(const std::vector<OpFuncNode>& vec_func_list,
                       std::vector<OperatorBase*>& op_list,  // NOLINT
                       const VariableScope& var_scope,
                       const platform::Place& place) {
  for (size_t i = 0; i < vec_func_list.size(); ++i) {
    auto& func_node = vec_func_list[i];
    auto op_base = op_list[i];
    // build runtime cost
    // VariableValueMap ins_map;
    std::vector<std::vector<Variable*>> ins_map;
    for (auto& var_name_item : func_node.input_name_map) {
      std::vector<Variable*> input_vars;

      input_vars.reserve(func_node.input_index[var_name_item.second].size());
      for (auto& id : func_node.input_index[var_name_item.second]) {
        // std::cerr << var_name_item.first << "\t " << id << std::endl;
        input_vars.emplace_back(var_scope.var_list[id].get());
      }
      // ins_map.emplace( var_name_item.first, std::move(input_vars) );
      ins_map.emplace_back(std::move(input_vars));
    }

    // VariableValueMap outs_map;
    std::vector<std::vector<Variable*>> outs_map;
    for (auto& var_name_item : func_node.output_name_map) {
      std::vector<Variable*> out_vars;

      out_vars.reserve(func_node.output_index[var_name_item.second].size());
      for (auto& id : func_node.output_index[var_name_item.second]) {
        // std::cerr << var_name_item.first << "\t " << id << std::endl;
        out_vars.emplace_back(var_scope.var_list[id].get());
      }
      // outs_map.emplace( var_name_item.first, std::move( out_vars ) );
      outs_map.emplace_back(std::move(out_vars));
    }

    RuntimeContextV2 runtime_context(
        ins_map, outs_map, func_node.input_name_map, func_node.output_name_map);
    // runtime_context.inputs.swap( ins_map );
    // runtime_context.outputs.swap(  outs_map );
    // runtime_context.input_values.swap(ins_map);
    // runtime_context.output_values.swap(outs_map);
    // runtime_context.input_name_map = func_node.input_name_map;
    // runtime_context.output_name_map = func_node.output_name_map;

    RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);

    // dynamic_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
    // &infer_shape_ctx );
    // RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
    static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape(
        &infer_shape_ctx);

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // auto place = platform::CPUPlace();
    // auto place = platform::CUDAPlace(0);
    auto* dev_ctx = pool.Get(place);
    Scope scope;

    auto exec_context =
        ExecutionContextV2(*op_base, scope, *dev_ctx, runtime_context);

    func_node.kernel_func_(exec_context);
  }
}

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place, const ProgramDesc& prog,
                  const ProgramDesc& startup_prog)
      : place_(place), prog_(prog) {
    paddle::framework::InitDevices();

    is_build = false;

    paddle::framework::build_variable_scope(startup_prog, &global_scope);

    std::vector<paddle::framework::OpFuncNode> vec_func_list;
    std::vector<paddle::framework::OperatorBase*> op_list;
    paddle::framework::build_op_func_list(startup_prog, op_list, vec_func_list,
                                          &global_scope, place_);
  }

  void run(const std::vector<std::string> vec_name,
           const std::vector<framework::Tensor>& vec_tensor,
           const std::vector<std::string>& vec_fetch_name,
           std::vector<framework::Tensor>& vec_out) {  // NOLINT
    // std::cerr << "run" << std::endl;
    // set static data
    if (is_build == false) {
      paddle::framework::build_variable_scope(prog_, &global_scope);
    }

    for (size_t i = 0; i < vec_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_name[i]);
      // std::cerr << "find " << (it != global_scope.name2id.end()) <<
      // std::endl;
      assert(it != global_scope.name2id.end());

      auto feed_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();
      // std::cerr << " get tensor" << std::endl;
      feed_tensor->ShareDataWith(vec_tensor[i]);
      // std::cerr << "share buffer with" << std::endl;
    }

    if (is_build == false) {
      paddle::framework::build_op_func_list(prog_, op_list, vec_func_list,
                                            &global_scope, place_);
      is_build = true;
    } else {
      paddle::framework::exec_op_func_list(vec_func_list, op_list, global_scope,
                                           place_);
    }

    for (size_t i = 0; i < vec_fetch_name.size(); ++i) {
      auto it = global_scope.name2id.find(vec_fetch_name[i]);
      assert(it != global_scope.name2id.end());

      auto fetch_tensor =
          global_scope.var_list[it->second]->GetMutable<framework::LoDTensor>();

      // std::cerr << "out  "  << fetch_tensor->data<float>()[0] << std::endl;
      if (platform::is_gpu_place(fetch_tensor->place())) {
        // std::cerr << "fetch gpu" << std::endl;
        Tensor out;
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        auto* dev_ctx = pool.Get(place_);
        dev_ctx->Wait();
        TensorCopySync(*fetch_tensor, platform::CPUPlace(), &out);
        dev_ctx->Wait();
        // std::cerr << "out  " << out << std::endl;
        vec_out.push_back(out);
      } else {
        // std::cerr << "out  " << *fetch_tensor << std::endl;
      }
    }
  }

 private:
  const platform::Place& place_;
  const ProgramDesc& prog_;
  paddle::framework::VariableScope global_scope;
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  std::vector<paddle::framework::OperatorBase*> op_list;

  bool is_build;
};
}  // namespace framework
}  // namespace paddle
