// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vitisai_execution_provider.h"

// Standard headers/libs.
#include <cassert>
#include <fstream>
#include <istream>
#include <filesystem>

// 1st-party headers/libs.
#include "core/platform/env_var_utils.h"
#include "core/common/exceptions.h"

#include "vaip/capability.h"
#include "vaip/global_api.h"
#include "ep_context_utils.h"

using namespace ONNX_NAMESPACE;

namespace fs = std::filesystem;

namespace onnxruntime {
constexpr const char* VITISAI = "VITISAI";

VitisAIExecutionProvider::VitisAIExecutionProvider(
    const ProviderOptions& info)
    // const ProviderOptions& info, const SessionOptions* p_sess_opts)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, info_(info) {
  CreateKernelRegistry();

  auto it = info_.find("ep_context_enable");
  ep_ctx_enabled_ = it != info_.end() && it->second == "1";
  it = info_.find("ep_context_embed_mode");
  ep_ctx_embed_mode_ = it != info_.end() && it->second != "0";
  // ep_ctx_embed_mode_ = it == info_.end() || it->second != "0";
  it = info_.find("ep_context_file_path");
  ep_ctx_model_path_cfg_ = it == info_.end() ? "" : it->second;
  LOGS_DEFAULT(VERBOSE) << "EP Context cache enabled: " << ep_ctx_enabled_;
  LOGS_DEFAULT(VERBOSE) << "EP context cache embed mode: " << ep_ctx_embed_mode_;
  LOGS_DEFAULT(VERBOSE) << "User specified EP context cache path: " << ep_ctx_model_path_cfg_;
}

void VitisAIExecutionProvider::CreateKernelRegistry() {
  for (const auto& domain : get_domains_vitisaiep()) {
    for (const auto* op : domain->custom_ops_) {
      vitisai_optypes_.insert(domain->domain_ + ":" + op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return get_kernel_registry_vitisaiep(); }

// This method is called after both `GetComputeCapabilityOps()` and `Compile()`.
// This timing is required to work with both compilation-based EPs and non-compilation-based EPs.
const InlinedVector<const Node*> VitisAIExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_node_ptrs;
  // All preconditions are supposed to have happened.
  if (p_ep_ctx_model_) {
    auto& graph = p_ep_ctx_model_->MainGraph();
    if (has_create_ep_context_nodes()) {
      auto nodes = create_ep_context_nodes(graph, **execution_providers_);
      if (nodes.has_value()) {
        ep_context_node_ptrs.assign(nodes->begin(), nodes->end());
      }
    } else {
      for (const auto* p_node : graph.Nodes()) {
        ep_context_node_ptrs.push_back(p_node);
      }
    }
  }
  return ep_context_node_ptrs;
}

void VitisAIExecutionProvider::LoadEPContexModelFromFile() const {
  // XXX: should "p_ep_ctx_model_" be checked or not?
  if (!p_ep_ctx_model_ && !ep_ctx_model_file_loc_.empty()) {
    auto status = Model::Load(ep_ctx_model_file_loc_, *p_ep_ctx_model_proto_);
    if (!status.IsOK()) {
      ORT_THROW("Loading EP context model failed from ", PathToUTF8String(ep_ctx_model_file_loc_));
    }
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
    LOGS_DEFAULT(VERBOSE) << "Loaded EP context model from: " << PathToUTF8String(ep_ctx_model_file_loc_);
  } else if (ep_ctx_model_file_loc_.empty()) {
    LOGS_DEFAULT(WARNING) << "Cannot load an EP-context model due to bad file path";
  }
}

void VitisAIExecutionProvider::PrepareEPContextEnablement(
    const onnxruntime::GraphViewer& graph_viewer) const {
  // Create a new model, reusing the graph name, the op-domain-to-opset-version map,
  // the op schema registry of the current graph, etc.
  p_ep_ctx_model_ = graph_viewer.CreateModel(*GetLogger());
  LOGS_DEFAULT(VERBOSE) << "Container model created";
}

std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer.GetGraph());
  // TODO: platform dependency (Linux vs Windows).
  model_path_str_ = ToPathString(GetTopLevelModelPath(graph_viewer).string());
  if (GetEPContextModelFileLocation(
          ep_ctx_model_path_cfg_, model_path_str_, is_ep_ctx_model, ep_ctx_model_file_loc_)) {
    if (is_ep_ctx_model) {
      LOGS_DEFAULT(VERBOSE) << "An EP context model passed in";
      LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << PathToUTF8String(ep_ctx_model_file_loc_);
    } else {
      if (fs::exists(ep_ctx_model_file_loc_) && fs::is_regular_file(ep_ctx_model_file_loc_) && ep_ctx_enabled_) {
        ORT_THROW("The inference session was created with a normal ONNX model but a model file with EP context cache exists at ",
                  PathToUTF8String(ep_ctx_model_file_loc_), ". Please remove the EP context model manually if you want to re-generate it.");
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to get EP context model file location";
  }

  if (graph_viewer.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  if (execution_providers_) {
    // Only compiling a model once is currently supported
    return {};
  }
  execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph_viewer, *GetLogger(), info_));
  auto result = vaip::GetComputeCapabilityOps(graph_viewer, execution_providers_.get(), vitisai_optypes_);
  size_t index = 0u;
  for (auto& ep : **execution_providers_) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(graph_viewer, ep.get(), index));
    index = index + 1;
  }
  if (ep_ctx_enabled_ && !is_ep_ctx_model) {
    PrepareEPContextEnablement(graph_viewer);
  }
  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    auto& attrs = fused_node_graph.fused_node.get().GetAttributes();
    assert(attrs.count("index"));
    size_t index = attrs.at("index").i();
    (**this->execution_providers_)[index]->set_fused_node(&fused_node_graph.fused_node.get());
    compute_info.create_state_func = [this, index](ComputeContext* context, FunctionState* state) {
      auto* p = (**this->execution_providers_)[index]->compile().release();
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

}  // namespace onnxruntime
