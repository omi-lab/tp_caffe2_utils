#include "tp_caffe2_utils/OpUtils.h"
#include "tp_caffe2_utils/FillOps.h"
#include "tp_caffe2_utils/ModelDetails.h"
#include "tp_caffe2_utils/Ops.h"
#include "tp_caffe2_utils/ArgUtils.h"
#include "tp_caffe2_utils/Print.h"

#include "tp_utils/DebugUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void removeOpByOutput(caffe2::NetDef& net,const std::string& opOutputName)
{
  for(auto i=net.mutable_op()->begin(); i!=net.mutable_op()->end(); ++i)
  {
    for(const auto& j : i->output())
    {
      if(j == opOutputName)
      {
        net.mutable_op()->erase(i);
        return;
      }
    }
  }
}

//##################################################################################################
void addGradientOps(std::vector<caffe2::OperatorDef*> gradientOps, caffe2::NetDef& trainNet)
{
  for(size_t i=gradientOps.size()-1; i<gradientOps.size(); i--)
  {
    auto op = gradientOps[i];

    std::vector<caffe2::GradientWrapper> output(size_t(op->output_size()));
    for(size_t j = 0; j < output.size(); j++)
      output[j].dense_ = op->output(int(j)) + "_grad";

    caffe2::GradientOpsMeta meta = caffe2::GetGradientForOp(*op, output);

    for(size_t j=0; j<meta.ops_.size(); j++)
    {
      auto grad = trainNet.add_op();
      grad->CopyFrom(meta.ops_[j]);
      grad->set_is_gradient_op(true);
    }
  }
}

//##################################################################################################
void addGradientOps(ModelDetails& model)
{
  for(const auto& subNet : model.trainSubNets)
    addGradientOps(subNet->gradientOps, subNet->trainNet);

  addGradientOps(model.gradientOps, model.trainNet);
}

//##################################################################################################
void addApplyGradientsOps_simple(ModelDetails& model, float lr)
{
  addConstantFillOp(model.initTrainNet, {1}, 1.0f, "one");
  addConstantFillOp(model.initTrainNet, {1}, lr, "lr");

  for(const auto& name : model.learntBlobNames)
    addWeightedSumOP(model.trainNet, name, "one", name + "_grad", "lr", name);
}

//##################################################################################################
void addApplyGradientsOps_momentum(ModelDetails& model, float lr, float momentum)
{
  addConstantFillOp(model.initTrainNet, {1}, 1.0f, "one");
  addConstantFillOp(model.initTrainNet, {1}, lr, "lr");
  addConstantFillOp(model.initTrainNet, {1}, momentum, "momentum");

  for(const auto& name : model.learntBlobNames)
  {
    tp_caffe2_utils::addConstantFillOp_copy(model.initTrainNet, name, 0.0f, name + "_momentum");

    addWeightedSumOP(model.trainNet, name              , "one"     , name + "_grad"    , "lr" , name              );
    addWeightedSumOP(model.trainNet, name              , "one"     , name + "_momentum", "lr" , name              );
    addWeightedSumOP(model.trainNet, name + "_momentum", "momentum", name + "_grad"    , "one", name + "_momentum");
  }
}

//##################################################################################################
void addApplyGradientsOps_clippedMomentum(ModelDetails& model,
                                          float lr,
                                          float momentum,
                                          float minGradient,
                                          float maxGradient)
{
  addConstantFillOp(model.initTrainNet, {1}, 1.0f, "one");
  addConstantFillOp(model.initTrainNet, {1}, lr, "lr");
  addConstantFillOp(model.initTrainNet, {1}, momentum, "momentum");

  for(const auto& name : model.learntBlobNames)
  {
    tp_caffe2_utils::addConstantFillOp_copy(model.initTrainNet, name, 0.0f, name + "_momentum");

    addClipOp(model.trainNet, name + "_grad", name + "_grad", minGradient, maxGradient);
    addWeightedSumOP(model.trainNet, name              , "one"     , name + "_grad"    , "lr" , name              );
    addWeightedSumOP(model.trainNet, name              , "one"     , name + "_momentum", "lr" , name              );
    addWeightedSumOP(model.trainNet, name + "_momentum", "momentum", name + "_grad"    , "one", name + "_momentum");
  }
}

//##################################################################################################
void addApplyGradientsOps_adamOptimizer(ModelDetails& model,
                                        float initialLR,
                                        int64_t initialIter,
                                        float beta1,
                                        float beta2,
                                        float epsilon)
{
  addConstantFillOp(model.initTrainNet, {1}, initialLR, "lr");
  model.cpuOps.insert(addConstantFillOp(model.initTrainNet, {1}, initialIter, "iter"));

  for(const auto& name : model.learntBlobNames)
  {
    tp_caffe2_utils::addConstantFillOp_copy(model.initTrainNet, name, 0.0f, name + "_moment_1");
    tp_caffe2_utils::addConstantFillOp_copy(model.initTrainNet, name, 0.0f, name + "_moment_2");

    auto op = model.trainNet.add_op();
    op->set_type("Adam");
    tp_caffe2_utils::addFloatArg(op, "beta1", beta1);
    tp_caffe2_utils::addFloatArg(op, "beta2", beta2);
    tp_caffe2_utils::addFloatArg(op, "epsilon", epsilon);

    op->add_input(name);
    op->add_input(name + "_moment_1");
    op->add_input(name + "_moment_2");
    op->add_input(name + "_grad");
    op->add_input("lr");
    op->add_input("iter");

    op->add_output(name);
    op->add_output(name + "_moment_1");
    op->add_output(name + "_moment_2");
  }
}
}
