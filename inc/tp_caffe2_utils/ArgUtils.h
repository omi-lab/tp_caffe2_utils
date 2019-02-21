#ifndef tp_caffe2_utils_ArgUtils_h
#define tp_caffe2_utils_ArgUtils_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void addIntArg(caffe2::OperatorDef* op, const std::string& name, int64_t value);

//##################################################################################################
void addIntsArg(caffe2::OperatorDef* op, const std::string& name, const std::vector<int64_t>& values);

//##################################################################################################
void addFloatArg(caffe2::OperatorDef* op, const std::string& name, float value);

//##################################################################################################
void addFloatsArg(caffe2::OperatorDef* op, const std::string& name, const std::vector<float>& values);

//##################################################################################################
void addStringArg(caffe2::OperatorDef* op, const std::string& name, const char* value);

//##################################################################################################
void addShapeArg(caffe2::OperatorDef* op, const std::vector<int64_t>& shape);

}

#endif
