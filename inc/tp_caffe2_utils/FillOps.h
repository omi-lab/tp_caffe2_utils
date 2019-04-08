#ifndef tp_caffe2_utils_FillOps_h
#define tp_caffe2_utils_FillOps_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
caffe2::OperatorDef* addXavierFillOp(caffe2::NetDef& net,
                                     const std::vector<int64_t>& shape,
                                     const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addMSRAFillOp(caffe2::NetDef& net,
                                   const std::vector<int64_t>& shape,
                                   const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addConstantFillOp(caffe2::NetDef& net,
                                       const std::vector<int64_t>& shape,
                                       float value,
                                       const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addConstantFillOp_copy(caffe2::NetDef& net,
                                            const std::string& copyShape,
                                            float value,
                                            const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addGaussianFillOp(caffe2::NetDef& net,
                                       const std::vector<int64_t>& shape,
                                       float mean,
                                       float sd,
                                       const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addGaussianFillOp_copy(caffe2::NetDef& net,
                                            const std::string& copyShape,
                                            float mean,
                                            float sd,
                                            const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addGivenTensorFillOp(caffe2::NetDef& net,
                                          const std::vector<int64_t>& shape,
                                          const std::vector<float>& values,
                                          const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addGivenTensorIntFillOp(caffe2::NetDef& net,
                                             const std::vector<int64_t>& shape,
                                             const std::vector<int64_t>& values,
                                             const std::string& output);

//##################################################################################################
caffe2::OperatorDef* addGivenTensorInt64FillOp(caffe2::NetDef& net,
                                               const std::vector<int64_t>& shape,
                                               const std::vector<int64_t>& values,
                                               const std::string& output);

}

#endif
