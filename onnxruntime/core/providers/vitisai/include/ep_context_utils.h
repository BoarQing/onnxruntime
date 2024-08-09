#pragma once

// Standard headers/libs.
#include <filesystem>
#include <vector>
#include <string>
#include <memory>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"

namespace fs = std::filesystem;

namespace onnxruntime {

static constexpr const char* kEPContextOp = "EPContext";
static constexpr const char* kSourceAttr = "source";

const Node* GetEPContextNodePtr(const Graph&);

void CreateEPContexNodes(Graph*, const std::vector<IExecutionProvider::FusedNodeAndGraph>&, const std::string&, const std::string&,
                         const int64_t, const std::string&, const std::string&, bool, const logging::Logger*);

bool GraphHasEPContextNode(const Graph&);

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>&);

const fs::path& GetTopLevelModelPath(const GraphViewer&);

bool GetEPContextModelFileLocation(
    const std::string&, const PathString&, bool, PathString&);

}  // namespace onnxruntime
