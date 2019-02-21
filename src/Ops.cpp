#include "tp_caffe2_utils/Ops.h"
#include "tp_caffe2_utils/FillOps.h"
#include "tp_caffe2_utils/ArgUtils.h"
#include "tp_caffe2_utils/ModelDetails.h"

namespace tp_caffe2_utils
{

//##################################################################################################
caffe2::OperatorDef* addActivationOp(caffe2::NetDef& net,
                                     const std::string& inName,
                                     const std::string& name,
                                     const std::string& function)
{
  auto op = net.add_op();
  op->set_type(function);
  op->add_input(inName);
  op->add_output(name);
  return op;
}

//##################################################################################################
caffe2::OperatorDef* addWeightedSumOP(caffe2::NetDef& net,
                                      const std::string& aName,
                                      const std::string& aWeightName,
                                      const std::string& bName,
                                      const std::string& bWeightName,
                                      const std::string& name)
{
  auto op = net.add_op();
  op->set_type("WeightedSum");
  op->add_input(aName);
  op->add_input(aWeightName);
  op->add_input(bName);
  op->add_input(bWeightName);
  op->add_output(name);
  return op;
}

//##################################################################################################
void addConv2DOp(ModelDetails& model,
               const std::string& inName,
               const std::string& name,
               int64_t inChannels,
               int64_t outChannels,
               int64_t stride,
               int64_t pad,
               int64_t kernelSize)
{
  model.dataBlobNames.push_back(name);

  auto op = model.predictNet.add_op();
  model.gradientOps.push_back(op);
  op->set_type("Conv2D");
  op->add_input(inName);
  op->add_input(name + "_filter");
  op->add_input(name + "_bias");
  op->add_output(name);

  addIntArg   (op, "stride", stride);
  addIntArg   (op, "pad"   , pad);
  addIntArg   (op, "kernel", kernelSize);
  addStringArg(op, "order" , "NCHW");

  addXavierFillOp  (model.initPredictNet, {outChannels, inChannels, kernelSize, kernelSize}, name + "_filter");
  addConstantFillOp(model.initPredictNet, {outChannels},                               0.0f, name + "_bias");

  model.learntBlobNames.push_back(name + "_filter");
  model.learntBlobNames.push_back(name + "_bias");
}

//##################################################################################################
caffe2::OperatorDef* addConcat(caffe2::NetDef& net,
                               const std::vector<std::string>& inNames,
                               const std::string& name,
                               const std::string& splitInfoName,
                               int64_t axis)
{
  auto op = net.add_op();
  op->set_type("Concat");
  addIntArg(op, "axis", axis);

  for(const auto& inName : inNames)
    op->add_input(inName);

  op->add_output(name);
  op->add_output(splitInfoName);

  return op;
}
}
