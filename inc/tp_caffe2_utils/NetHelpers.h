#ifndef tp_caffe2_utils_NetHelpers_h
#define tp_caffe2_utils_NetHelpers_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
//! If CUDA is available use that else use CPU
void setDeviceType(caffe2::NetDef& net);

//##################################################################################################
void setDeviceType(caffe2::NetDef& net, int32_t value);

}

#endif
