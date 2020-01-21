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
void addConv2DOp(ModelDetails& model,
                 const std::string& inName,
                 const std::string& name,
                 int64_t inChannels,
                 int64_t outChannels,
                 int64_t strideW,
                 int64_t strideH,
                 int64_t padT,
                 int64_t padL,
                 int64_t padB,
                 int64_t padR,
                 int64_t kernelW,
                 int64_t kernelH)
{
  auto op = model.predictNet.add_op();
  model.gradientOps.push_back(op);
  op->set_type("Conv2D");
  op->add_input(inName);
  op->add_input(name + "_filter");
  op->add_input(name + "_bias");
  op->add_output(name);

  addIntArg   (op, "stride_w", strideW);
  addIntArg   (op, "stride_h", strideH);
  addIntArg   (op, "pad_t"   , padT);
  addIntArg   (op, "pad_l"   , padL);
  addIntArg   (op, "pad_b"   , padB);
  addIntArg   (op, "pad_r"   , padR);
  addIntArg   (op, "kernel_w"   , kernelW);
  addIntArg   (op, "kernel_h"   , kernelH);
  addStringArg(op, "order" , "NCHW");

  addXavierFillOp  (model.initPredictNet, {outChannels, inChannels, kernelH, kernelW}, name + "_filter");
  addConstantFillOp(model.initPredictNet, {outChannels},                               0.0f, name + "_bias");

  model.learntBlobNames.push_back(name + "_filter");
  model.learntBlobNames.push_back(name + "_bias");
}

//##################################################################################################
void addAveragePool2DOp(ModelDetails& model,
                        const std::string& inName,
                        const std::string& name,
                        int64_t strideW,
                        int64_t strideH,
                        int64_t padT,
                        int64_t padL,
                        int64_t padB,
                        int64_t padR,
                        int64_t kernelW,
                        int64_t kernelH)
{
  auto op = model.predictNet.add_op();
  model.gradientOps.push_back(op);
  op->set_type("AveragePool2D");
  op->add_input(inName);
  op->add_output(name);

  addIntArg   (op, "stride_w", strideW);
  addIntArg   (op, "stride_h", strideH);
  addIntArg   (op, "pad_t"   , padT);
  addIntArg   (op, "pad_l"   , padL);
  addIntArg   (op, "pad_b"   , padB);
  addIntArg   (op, "pad_r"   , padR);
  addIntArg   (op, "kernel_w"   , kernelW);
  addIntArg   (op, "kernel_h"   , kernelH);
  addStringArg(op, "order" , "NCHW");
}

//##################################################################################################
caffe2::OperatorDef* addConcatOp(caffe2::NetDef& net,
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


//##################################################################################################
caffe2::OperatorDef* addClipOp(caffe2::NetDef& net,
                               const std::string& inName,
                               const std::string& name,
                               float min,
                               float max)
{
  auto op = net.add_op();
  op->set_type("Clip");
  tp_caffe2_utils::addFloatArg(op, "min", min);
  tp_caffe2_utils::addFloatArg(op, "max", max);
  op->add_input(inName);
  op->add_output(name);
  return op;
}

//##################################################################################################
caffe2::OperatorDef* addMathOp(caffe2::NetDef& net,
                               const std::string& aName,
                               const std::string& bName,
                               const std::string& name,
                               const std::string& function)
{
  auto op = net.add_op();
  op->set_type(function);
  op->add_input(aName);
  op->add_input(bName);
  op->add_output(name);
  return op;
}

//##################################################################################################
caffe2::OperatorDef* addFCOp(ModelDetails& model,
                             const std::string inName,
                             const std::string outName,
                             int64_t inSize,
                             int64_t outSize)
{
  auto op = model.predictNet.add_op();
  op->set_type("FC");
  op->add_input(inName);
  op->add_input(outName+"_weights");
  op->add_input(outName+"_bias");
  op->add_output(outName);

  tp_caffe2_utils::addXavierFillOp  (model.initPredictNet, {outSize, inSize}, outName+"_weights"); // {outFloats, inFloats}
  tp_caffe2_utils::addConstantFillOp(model.initPredictNet, {outSize},   0.0f, outName+"_bias"   ); // {outFloats}, initialValue

  model.learntBlobNames.push_back(outName+"_weights");
  model.learntBlobNames.push_back(outName+"_bias");

  return op;
}

//##################################################################################################
void addFCActivationOps(ModelDetails& model,
                        std::vector<caffe2::OperatorDef*>& gradientOps,
                        const std::string inName,
                        const std::string outName,
                        int64_t inSize,
                        int64_t outSize,
                        const std::string& function)
{
  auto fcName = outName+"_fc";
  gradientOps.push_back(addFCOp(model, inName, fcName, inSize, outSize));
  gradientOps.push_back(tp_caffe2_utils::addActivationOp(model.predictNet, fcName, outName, function));
}

//##################################################################################################
caffe2::OperatorDef* addDropoutOp(ModelDetails& model,
                                  const std::string inName,
                                  const std::string outName,
                                  float ratio,
                                  bool dropout)
{
  auto op = model.predictNet.add_op();
  op->set_type("Dropout");
  tp_caffe2_utils::addFloatArg(op, "ratio", ratio);
  tp_caffe2_utils::addIntArg(op, "is_test", dropout?0:1);
  op->add_input(inName);
  op->add_output(outName);
  op->add_output(outName + "_mask");
  return op;
}

}
