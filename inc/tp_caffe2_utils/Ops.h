#ifndef tp_caffe2_utils_Ops_h
#define tp_caffe2_utils_Ops_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

struct ModelDetails;

//##################################################################################################
caffe2::OperatorDef* addActivationOp(caffe2::NetDef& net,
                                     const std::string& inName,
                                     const std::string& name,
                                     const std::string& function);

//##################################################################################################
caffe2::OperatorDef* addWeightedSumOP(caffe2::NetDef& net,
                                      const std::string& aName,
                                      const std::string& aWeightName,
                                      const std::string& bName,
                                      const std::string& bWeightName,
                                      const std::string& name);

//##################################################################################################
void addConv2DOp(ModelDetails& model,
               const std::string& inName,
               const std::string& name,
               int64_t inChannels,
               int64_t outChannels,
               int64_t stride,
               int64_t pad,
               int64_t kernelSize);

//##################################################################################################
caffe2::OperatorDef* addConcat(caffe2::NetDef& net,
                               const std::vector<std::string>& inNames,
                               const std::string& name,
                               const std::string& splitInfoName,
                               int64_t axis = 1);

}

#endif
