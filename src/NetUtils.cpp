#include "tp_caffe2_utils/NetUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void setDeviceType(caffe2::NetDef& net)
{
#ifdef TP_CUDA
  setDeviceType(net, caffe2::PROTO_CUDA);
    #else
  setDeviceType(net, caffe2::PROTO_CPU);
#endif
}

//##################################################################################################
void setDeviceType(caffe2::NetDef& net, int32_t value)
{
  net.mutable_device_option()->set_device_type(value);
  for(int i=0; i<net.op_size(); i++)
    net.mutable_op(i)->mutable_device_option()->set_device_type(value);
}

}
