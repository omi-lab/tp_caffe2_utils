#ifndef tp_caffe2_utils_FillOps_h
#define tp_caffe2_utils_FillOps_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void addXavierFillOp(caffe2::NetDef& net,
                     const std::vector<int64_t>& shape,
                     const std::string& output);

//##################################################################################################
void addMSRAFillOp(caffe2::NetDef& net,
                   const std::vector<int64_t>& shape,
                   const std::string& output);

//##################################################################################################
void addConstantFillOp(caffe2::NetDef& net,
                       const std::vector<int64_t>& shape,
                       float value,
                       const std::string& output);

//##################################################################################################
void addCopyConstantFillOp(caffe2::NetDef& net,
                           const std::string& copyShape,
                           float value,
                           const std::string& output);

//##################################################################################################
void addGaussianFillOp(caffe2::NetDef& net,
                       const std::vector<int64_t>& shape,
                       float mean,
                       float sd,
                       const std::string& output);

//##################################################################################################
void addGivenTensorFillOp(caffe2::NetDef& net,
                          const std::vector<int64_t>& shape,
                          const std::vector<float>& values,
                          const std::string& output);

//##################################################################################################
void addGivenTensorIntFillOp(caffe2::NetDef& net,
                             const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& values,
                             const std::string& output);

//##################################################################################################
void addGivenTensorInt64FillOp(caffe2::NetDef& net,
                               const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& values,
                               const std::string& output);

}

#endif
