#ifndef tp_caffe2_utils_Print_h
#define tp_caffe2_utils_Print_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void printBlob(caffe2::Workspace& workspace, const std::string& name);

//##################################################################################################
void printAllBlobShapes(caffe2::Workspace& workspace);

//##################################################################################################
void printArg(const caffe2::Argument& arg);

//##################################################################################################
void printOp(const caffe2::OperatorDef& op);

//##################################################################################################
void printOps(const caffe2::NetDef& net);

}

#endif
