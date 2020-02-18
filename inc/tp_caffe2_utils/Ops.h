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
void addConv2DOp(ModelDetails& model,
                 const std::string& inName,
                 const std::string& name,
                 int64_t inChannels,
                 int64_t outChannels,
                 int64_t strideW,
                 int64_t strideH,
                 int64_t padT,
                 int64_t padL,
                 int64_t padB,
                 int64_t padR,
                 int64_t kernelW,
                 int64_t kernelH);

//##################################################################################################
void addConv2DActivationOps(ModelDetails& model,
                            const std::string& inName,
                            const std::string& outName,
                            int64_t inChannels,
                            int64_t outChannels,
                            int64_t stride,
                            int64_t pad,
                            int64_t kernelSize,
                            const std::string& function);

//##################################################################################################
void addConv2DActivationOps(ModelDetails& model,
                            const std::string& inName,
                            const std::string& outName,
                            int64_t inChannels,
                            int64_t outChannels,
                            int64_t strideW,
                            int64_t strideH,
                            int64_t padT,
                            int64_t padL,
                            int64_t padB,
                            int64_t padR,
                            int64_t kernelW,
                            int64_t kernelH,
                            const std::string& function);

//##################################################################################################
void addAveragePool2DOp(ModelDetails& model,
                        const std::string& inName,
                        const std::string& name,
                        int64_t strideW,
                        int64_t strideH,
                        int64_t padT,
                        int64_t padL,
                        int64_t padB,
                        int64_t padR,
                        int64_t kernelW,
                        int64_t kernelH);

//##################################################################################################
caffe2::OperatorDef* addConcatOp(caffe2::NetDef& net,
                                 const std::vector<std::string>& inNames,
                                 const std::string& name,
                                 const std::string& splitInfoName,
                                 int64_t axis = 1);

//##################################################################################################
caffe2::OperatorDef* addClipOp(caffe2::NetDef& net,
                               const std::string& inName,
                               const std::string& name,
                               float min,
                               float max);

//##################################################################################################
caffe2::OperatorDef* addMathOp(caffe2::NetDef& net,
                               const std::string& aName,
                               const std::string& bName,
                               const std::string& name,
                               const std::string& function);

//##################################################################################################
caffe2::OperatorDef* addFCOp(ModelDetails& model,
                             const std::string inName,
                             const std::string outName,
                             int64_t inSize,
                             int64_t outSize);

//##################################################################################################
void addFCActivationOps(ModelDetails& model,
                        std::vector<caffe2::OperatorDef*>& gradientOps,
                        const std::string inName,
                        const std::string outName,
                        int64_t inSize,
                        int64_t outSize,
                        const std::string& function);

//##################################################################################################
caffe2::OperatorDef* addDropoutOp(ModelDetails& model,
                                  const std::string inName,
                                  const std::string outName,
                                  float ratio,
                                  bool dropout);



}

#endif
