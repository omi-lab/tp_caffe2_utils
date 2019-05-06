#include "tp_caffe2_utils/NetUtils.h"
#include "tp_caffe2_utils/ModelDetails.h"

namespace tp_caffe2_utils
{
//##################################################################################################
void setDeviceType(ModelDetails& model)
{
  setDeviceType(model.initPredictNet, model.cpuOps);
  setDeviceType(model.initTrainNet, model.cpuOps);
  setDeviceType(model.predictNet, model.cpuOps);
  setDeviceType(model.trainNet, model.cpuOps);
}

//##################################################################################################
void setDeviceType(caffe2::NetDef& net, const std::unordered_set<caffe2::OperatorDef*>& cpuOps)
{
#ifdef TP_CUDA
  setDeviceType(net, cpuOps, caffe2::PROTO_CUDA);
    #else
  setDeviceType(net, cpuOps, caffe2::PROTO_CPU);
#endif
}

//##################################################################################################
void setDeviceType(caffe2::NetDef& net, const std::unordered_set<caffe2::OperatorDef*>& cpuOps, int32_t value)
{
  net.mutable_device_option()->set_device_type(value);
  for(int i=0; i<net.op_size(); i++)
  {
    auto op = net.mutable_op(i);
    auto v = tpContains(cpuOps, op)?caffe2::PROTO_CPU:value;
    op->mutable_device_option()->set_device_type(v);
  }
}

}
