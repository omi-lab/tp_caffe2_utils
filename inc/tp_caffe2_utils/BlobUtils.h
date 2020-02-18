#ifndef tp_caffe2_utils_BlobUtils_h
#define tp_caffe2_utils_BlobUtils_h

#include "tp_utils/DebugUtils.h"

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
std::vector<int64_t> blobDims(caffe2::Workspace& workspace,
                              const std::string& name);

//##################################################################################################
float readSingleValue(caffe2::Workspace& workspace,
                      const std::string& name);

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
template<typename T>
bool setBlob(caffe2::Workspace& workspace,
             const std::string& name,
             const std::vector<T>& inputData)
{
  caffe2::Blob* blob = workspace.GetBlob(name);
  if(!blob)
  {
    tpWarning() << "Failed to find " << name << " blob.";
    return false;
  }

  const auto& tensor = blob->GetMutable<caffe2::TensorCPU>();
  if(tensor->size() != int(inputData.size()))
  {
    tpWarning() << "Failed to set " << name << " blob.";
    tpWarning() << "Blob size: " << tensor->size() << " data size: " << inputData.size();
    return false;
  }

  tensor->CopyFrom(caffe2::TensorCPUFromValues<T>(tensorDims(*tensor), inputData));

  return true;
}

//##################################################################################################
bool setBlob(caffe2::Workspace& workspace,
             const std::string& name,
             const std::vector<float>& inputData,
             const std::vector<int64_t>& blobDims);

//##################################################################################################
bool setLR(caffe2::Workspace& workspace,
           float lr);

//##################################################################################################
template<typename T>
bool setSingleValue(caffe2::Workspace& workspace,
                    const std::string& name,
                    T value)
{
  std::vector<T> val;
  val.push_back(value);
  return setBlob(workspace, name, val);
}

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
