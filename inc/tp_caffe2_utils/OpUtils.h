#ifndef tp_caffe2_utils_OpUtils_h
#define tp_caffe2_utils_OpUtils_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void removeOpByOutput(caffe2::NetDef& net,const std::string& opOutputName);

}

#endif
