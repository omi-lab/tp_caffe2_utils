#include "tp_caffe2_utils/FillOps.h"
#include "tp_caffe2_utils/ArgUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void addXavierFillOp(caffe2::NetDef& net,
                     const std::vector<int64_t>& shape,
                     const std::string& output)
{
  auto op = net.add_op();
  op->set_type("XavierFill");
  addShapeArg(op, shape);
  op->add_output(output);
}

//##################################################################################################
void addMSRAFillOp(caffe2::NetDef& net,
                   const std::vector<int64_t>& shape,
                   const std::string& output)
{
  auto op = net.add_op();
  op->set_type("MSRAFill");
  addShapeArg(op, shape);
  op->add_output(output);
}

//##################################################################################################
void addConstantFillOp(caffe2::NetDef& net,
                       const std::vector<int64_t>& shape,
                       float value,
                       const std::string& output)
{
  auto op = net.add_op();
  op->set_type("ConstantFill");
  addShapeArg(op, shape);
  addFloatArg(op, "value", value);
  op->add_output(output);
}

//##################################################################################################
void addCopyConstantFillOp(caffe2::NetDef& net,
                           const std::string& copyShape,
                           float value,
                           const std::string& output)
{
  auto op = net.add_op();
  op->set_type("ConstantFill");
  addFloatArg(op, "value", value);
  op->add_input(copyShape);
  op->add_output(output);
}

//##################################################################################################
void addGaussianFillOp(caffe2::NetDef& net,
                       const std::vector<int64_t>& shape,
                       float mean,
                       float sd,
                       const std::string& output)
{
  auto op = net.add_op();
  op->set_type("GaussianFill");
  addShapeArg(op, shape);
  addFloatArg(op, "mean", mean);
  addFloatArg(op, "std", sd);
  op->add_output(output);
}

//##################################################################################################
void addGivenTensorFillOp(caffe2::NetDef& net,
                          const std::vector<int64_t>& shape,
                          const std::vector<float>& values,
                          const std::string& output)
{
  auto op = net.add_op();
  op->set_type("GivenTensorFill");
  addShapeArg(op, shape);
  addFloatsArg(op, "values", values);
  op->add_output(output);
}

//##################################################################################################
void addGivenTensorIntFillOp(caffe2::NetDef& net,
                             const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& values,
                             const std::string& output)
{
  auto op = net.add_op();
  op->set_type("GivenTensorIntFill");
  addShapeArg(op, shape);
  addIntsArg(op, "values", values);
  op->add_output(output);
}

//##################################################################################################
void addGivenTensorInt64FillOp(caffe2::NetDef& net,
                               const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& values,
                               const std::string& output)
{
  auto op = net.add_op();
  op->set_type("GivenTensorInt64Fill");
  addShapeArg(op, shape);
  addIntsArg(op, "values", values);
  op->add_output(output);
}

}
