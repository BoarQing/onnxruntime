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

constexpr const uint8_t kXCCode = 1;
[[maybe_unused]] constexpr const uint8_t kDDCode = 2;
[[maybe_unused]] constexpr const uint8_t kVCode = 4;

static constexpr const char* kEPContextOp = "EPContext";
static constexpr const char* kMainContextAttr = "main_context";
static constexpr const char* kEPCacheContextAttr = "ep_cache_context";
static constexpr const char* kEmbedModeAttr = "embed_mode";
static constexpr const char* kPartitionNameAttr = "partition_name";
static constexpr const char* kSourceAttr = "source";
static constexpr const char* kEPSDKVersionAttr = "ep_sdk_version";
static constexpr const char* kONNXModelFileNameAttr = "onnx_model_filename";
static constexpr const char* kNotesAttr = "notes";
static constexpr const char* kEPContextOpDomain = "com.microsoft";
static constexpr const char* kEPContextOpName = "VitisAIEPContextOp";

const Node* GetEPContextNodePtr(const Graph&);

void CreateEPContexNodes(Graph*, const std::vector<IExecutionProvider::FusedNodeAndGraph>&, const std::string&, const std::string&,
                         const int64_t, const std::string&, const std::string&, bool, const logging::Logger*);

bool GraphHasEPContextNode(const Graph&);

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>&);

const fs::path& GetTopLevelModelPath(const GraphViewer&);

bool GetEPContextModelFileLocation(
    const std::string&, const PathString&, bool, PathString&);

// The file for EP context cache is in the same folder as the EP context model file.
PathString GetEPContextCacheFileLocation(const PathString&, const PathString&);

}  // namespace onnxruntime
