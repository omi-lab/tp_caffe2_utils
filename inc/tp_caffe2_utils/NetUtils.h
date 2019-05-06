#ifndef tp_caffe2_utils_NetUtils_h
#define tp_caffe2_utils_NetUtils_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{
struct ModelDetails;

//##################################################################################################
//! If CUDA is available use that else use CPU
void setDeviceType(ModelDetails& model);

//##################################################################################################
//! If CUDA is available use that else use CPU
void setDeviceType(caffe2::NetDef& net, const std::unordered_set<caffe2::OperatorDef*>& cpuOps);

//##################################################################################################
void setDeviceType(caffe2::NetDef& net, const std::unordered_set<caffe2::OperatorDef*>& cpuOps, int32_t value);

}

#endif
