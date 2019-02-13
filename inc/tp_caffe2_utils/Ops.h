#ifndef tp_caffe2_utils_Ops_h
#define tp_caffe2_utils_Ops_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

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

}

#endif
