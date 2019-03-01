#ifndef tp_caffe2_utils_BlobUtils_h
#define tp_caffe2_utils_BlobUtils_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
struct Blob
{
  std::string name;
  std::vector<float> data;
  std::vector<int64_t> shape;
};

//##################################################################################################
struct ModelWeights
{
  std::vector<std::shared_ptr<Blob>> blobs;
};

//##################################################################################################
std::vector<int64_t> tensorDims(const caffe2::TensorCPU& tensor);

//##################################################################################################
void readBlob(caffe2::Workspace& workspace,
              const std::string& name,
              std::vector<float>& blobData);

//##################################################################################################
void readBlob(caffe2::Workspace& workspace,
              const std::string& name,
              std::vector<float>& blobData,
              std::vector<int64_t>& blobDims);

//##################################################################################################
bool setBlob(caffe2::Workspace& workspace,
             const std::string& name,
             const std::vector<float>& inputData);

//##################################################################################################
bool setBlob(caffe2::Workspace& workspace,
             const std::string& name,
             const std::vector<float>& inputData,
             const std::vector<int64_t>& blobDims);

//##################################################################################################
bool setLR(caffe2::Workspace& workspace,
           float lr);

//##################################################################################################
ModelWeights* saveModelWeights(caffe2::Workspace& workspace,
                               const std::vector<std::string>& learntBlobNames);

//##################################################################################################
bool loadModelWeights(caffe2::Workspace& workspace,
                      const ModelWeights* modelWeights,
                      std::string& error);

//##################################################################################################
std::string saveWeights(caffe2::Workspace& workspace,
                        const std::vector<std::string>& learntBlobNames);

//##################################################################################################
bool loadWeights(caffe2::Workspace& workspace,
                 const std::vector<std::string>& learntBlobNames,
                 const std::string& modelData,
                 std::string& error);

//##################################################################################################
void extractGivenTensorFill(const caffe2::NetDef& net,
                            ModelWeights& modelWeights);

}

#endif
