#include "tp_caffe2_utils/Ops.h"

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

}
