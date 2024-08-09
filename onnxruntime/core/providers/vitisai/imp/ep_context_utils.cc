// Standard headers/libs.
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include "ep_context_utils.h"

namespace onnxruntime {

constexpr const char* kVitisAI = "vitisai";

const Node* GetEPContextNodePtr(const Graph& graph) {
  // TODO: Support for multi-node EP context model.
  for (const auto* p_node : graph.Nodes()) {
    if (p_node->OpType() == kEPContextOp) {
      return p_node;
    }
  }
  return nullptr;
}

bool GraphHasEPContextNode(const Graph& graph) {
  size_t vitisai_len = std::strlen(kVitisAI);
  for (const auto* p_node : graph.Nodes()) {
    if (p_node->OpType() != kEPContextOp) {
      continue;
    }
    const auto& attrs = p_node->GetAttributes();
    if (attrs.count(kSourceAttr) == 0) {
      continue;
    }
    const auto& source_val = attrs.at(kSourceAttr).s();
    if (source_val == kVitisAIExecutionProvider) {
      return true;
    }
    if (source_val.length() != vitisai_len) {
      continue;
    }
    size_t j = 0;
    do {
      if (static_cast<unsigned char>(std::tolower(source_val[j])) != kVitisAI[j]) {
        break;
      }
      ++j;
    } while (j < vitisai_len);
    if (j == vitisai_len) {
      return true;
    }
  }
  return false;
}

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    bool has_node = GraphHasEPContextNode(fused_node_graph.filtered_graph.get().GetGraph());
    if (has_node) {
      return true;
    }
  }
  return false;
}

const fs::path& GetTopLevelModelPath(const GraphViewer& graph_viewer) {
  const auto& graph = graph_viewer.GetGraph();
  const Graph* p_graph = &graph;
  while (p_graph->IsSubgraph()) {
    p_graph = p_graph->ParentGraph();
  }
  return p_graph->ModelPath();
}

bool GetEPContextModelFileLocation(
    const std::string& ep_ctx_model_path_cfg,
    const PathString& model_path_str,
    bool is_ep_ctx_model,
    PathString& ep_ctx_model_file_loc) {
  if (!ep_ctx_model_file_loc.empty()) {
    return true;
  }
  if (!ep_ctx_model_path_cfg.empty()) {
    ep_ctx_model_file_loc = ToPathString(ep_ctx_model_path_cfg);
  } else if (!model_path_str.empty()) {
    if (is_ep_ctx_model) {
      ep_ctx_model_file_loc = model_path_str;
    } else {
      // Two alternatives for this case.
      // Alternative 1:
      // 1) Implement/override the method `IExecutionProvider::GetEpContextNodes()`.
      // 2) And follow how the default path is implemented in `CreateEpContextModel()`
      // in the file "graph_partitioner.cc".
      // 3) Model dump is not required.
      // Alternative 2:
      // 1) Do NOT implement/override `IExecutionProvider::GetEpContextNodes()`.
      // 2) No need to follow `CreateEpContextModel()` in the file "graph_partitioner.cc",
      // freely implement what the default path is like.
      // 3) Model dump is required.
#if 0
      ep_ctx_model_file_loc = model_path_str + ToPathString("_ctx.onnx");
#endif
#if 1
      fs::path model_fs_path(model_path_str);
      fs::path ep_ctx_model_fs_path(model_fs_path.parent_path() / model_fs_path.stem());
      ep_ctx_model_fs_path += fs::path("_ctx.onnx");
      ep_ctx_model_file_loc = ToPathString(ep_ctx_model_fs_path.string());
#endif
    }
  }
  return !ep_ctx_model_file_loc.empty();
}

}  // namespace onnxruntime
