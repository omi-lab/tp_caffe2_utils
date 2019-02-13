#include "tp_caffe2_utils/ArgUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void addIntArg(caffe2::OperatorDef* op, const std::string& name, int64_t value)
{
  auto arg = op->add_arg();
  arg->set_name(name);
  arg->set_i(value);
}

//##################################################################################################
void addIntsArg(caffe2::OperatorDef* op, const std::string& name, const std::vector<int64_t>& values)
{
  auto arg = op->add_arg();
  arg->set_name(name);
  for(auto v : values)
    arg->add_ints(v);
}

//##################################################################################################
void addFloatArg(caffe2::OperatorDef* op, const std::string& name, float value)
{
  auto arg = op->add_arg();
  arg->set_name(name);
  arg->set_f(value);
}

//##################################################################################################
void addFloatsArg(caffe2::OperatorDef* op, const std::string& name, const std::vector<float>& values)
{
  auto arg = op->add_arg();
  arg->set_name(name);
  for(auto v : values)
    arg->add_floats(v);
}

//##################################################################################################
void addStringArg(caffe2::OperatorDef* op, const std::string& name, const char* value)
{
  auto arg = op->add_arg();
  arg->set_name(name);
  arg->set_s(value);
}

//##################################################################################################
void addShapeArg(caffe2::OperatorDef* op, const std::vector<int64_t>& shape)
{
  addIntsArg(op, "shape", shape);
}

}
