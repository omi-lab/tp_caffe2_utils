#include "tp_caffe2_utils/BlobUtils.h"

#include "tp_utils/JSONUtils.h"

#include "json.hpp"

namespace tp_caffe2_utils
{

//##################################################################################################
std::vector<int64_t> tensorDims(const caffe2::TensorCPU& tensor)
{
  std::vector<int64_t> blobDims;
#if 1
  for(int i=0; i<tensor.ndim(); i++)
    blobDims.push_back(tensor.dim(i));
#else
  for(auto dim : tensor.dims())
    blobDims.push_back(dim);
#endif
  return blobDims;
}

//##################################################################################################
std::vector<int64_t> blobDims(caffe2::Workspace& workspace,
                              const std::string& name)
{
  std::vector<float> blobData;
  std::vector<int64_t> blobDims;
  readBlob(workspace, name, blobData, blobDims);
  return blobDims;
}

//##################################################################################################
void readBlob(caffe2::Workspace& workspace,
              const std::string& name,
              std::vector<float>& blobData)
{
  std::vector<int64_t> blobDims;
  readBlob(workspace, name, blobData, blobDims);
}

//##################################################################################################
void readBlob(caffe2::Workspace& workspace,
              const std::string& name,
              std::vector<float>& blobData,
              std::vector<int64_t>& blobDims)
{
  caffe2::Blob* blob = workspace.GetBlob(name);
  if(!blob)
    return;

#ifdef TP_CUDA
  if(caffe2::BlobIsTensorType(*blob, caffe2::CUDA))
  {
    caffe2::Blob* blobCPU = workspace.CreateBlob(name+"CPU");
    auto* tensorGPU = BlobGetMutableTensor(blob, caffe2::CUDA);

    auto* tensorCPU = caffe2::BlobGetMutableTensor(blobCPU, caffe2::CPU);
    tensorCPU->CopyFrom(*tensorGPU);

    const auto &data = tensorCPU->data<float>();
    blobData = std::vector<float>(data, data + tensorCPU->size());

    //for(auto dim : tensorCPU->dims())
    //  blobDims.push_back(dim);
    for(int i = 0; i < tensorCPU->ndim(); i++)
      blobDims.push_back(tensorCPU->dim(i));
  }
  else
#endif
  {
    if(blob->IsType<caffe2::TensorCPU>())
    {
      const auto& tensor = blob->Get<caffe2::TensorCPU>();

      if(tensor.IsType<float>())
      {
        const auto& data = tensor.data<float>();
        blobData = std::vector<float>(data, data + tensor.size());
        blobDims = tensorDims(tensor);
      }
    }
  }
}

//##################################################################################################
bool setBlob(caffe2::Workspace& workspace,
             const std::string& name,
             const std::vector<float>& inputData,
             const std::vector<int64_t>& blobDims)
{
  caffe2::Blob* blob = workspace.CreateBlob(name);
  if(!blob)
  {
    tpWarning() << "Failed to create " << name << " blob.";
    return false;
  }

  caffe2::BlobSetTensor(blob, caffe2::TensorCPUFromValues<float>(blobDims, inputData));
  return true;
}

//##################################################################################################
bool setLR(caffe2::Workspace& workspace, float lr)
{
  return setSingleValue(workspace, "lr", lr);
}

//##################################################################################################
ModelWeights* saveModelWeights(caffe2::Workspace& workspace,
                               const std::vector<std::string>& learntBlobNames)
{
  auto modelWeights = new ModelWeights;
  for(const auto& name : learntBlobNames)
  {
    auto blob = modelWeights->blobs.emplace_back(new Blob);
    blob->name = name;
    std::vector<int64_t> blobDims;
    readBlob(workspace, name, blob->data, blobDims);
  }
  return modelWeights;
}

//##################################################################################################
bool loadModelWeights(caffe2::Workspace& workspace,
                      const ModelWeights* modelWeights,
                      std::string& error)
{
  TP_UNUSED(error);

  for(const auto& blob : modelWeights->blobs)
    setBlob(workspace, blob->name, blob->data);

  return true;
}

//##################################################################################################
std::string saveWeights(caffe2::Workspace& workspace,
                        const std::vector<std::string>& learntBlobNames)
{
  nlohmann::json j;
  for(const auto& name : learntBlobNames)
  {
    nlohmann::json& jj = j[name];

    std::vector<float> blobData;
    std::vector<int64_t> blobDims;
    readBlob(workspace, name, blobData, blobDims);

    {
      nlohmann::json& dims = jj["dims"];
      dims = nlohmann::json::array();
      for(auto dim : blobDims)
        dims.push_back(dim);
    }

    {
      nlohmann::json& values = jj["values"];
      values = nlohmann::json::array();
      const float* d = blobData.data();
      const float* dMax = d+blobData.size();
      for(; d<dMax; d++)
        values.push_back(*d);
    }
  }
  return j.dump();
}

//##################################################################################################
bool loadWeights(caffe2::Workspace& workspace,
                 const std::vector<std::string>& learntBlobNames,
                 const std::string& modelData,
                 std::string& error)
{
  try
  {
    nlohmann::json j = nlohmann::json::parse(modelData);
    for(const std::string& name : learntBlobNames)
    {
      nlohmann::json jj = TPJSON(j, name);
      {
        std::vector<int64_t> dims = jj["dims"];
        std::vector<float> values = jj["values"];
        tpWarning() << "dims:" << dims.size() << " values:" << values.size();
        setBlob(workspace, name, values);
      }
    }
  }
  catch(...)
  {
    error = "Failed to parse model weights.";
    return false;
  }

  return true;
}

//##################################################################################################
void extractGivenTensorFill(const caffe2::NetDef& net, ModelWeights& modelWeights)
{
  for(int i=0; i<net.op_size(); i++)
  {
    const auto& op = net.op(i);
    if(op.type() != "GivenTensorFill")
      continue;

    if(op.output_size()<1)
      continue;

    auto& result = modelWeights.blobs.emplace_back(new Blob);
    result->name = op.output(0);

    for(int i=0; i<op.arg_size(); i++)
    {
      const auto& arg = op.arg(i);

      if(arg.name() == "values")
      {
        result->data.resize(size_t(arg.floats_size()));
        for(int j=0; j<arg.floats_size(); j++)
          result->data[size_t(j)] = arg.floats(j);
      }

      else if(arg.name() == "shape")
      {
        result->shape.resize(size_t(arg.ints_size()));
        for(int j=0; j<arg.ints_size(); j++)
          result->shape[size_t(j)] = arg.ints(j);
      }
    }
  }
}

}
