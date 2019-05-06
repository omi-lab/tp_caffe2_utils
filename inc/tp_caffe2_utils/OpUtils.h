#ifndef tp_caffe2_utils_OpUtils_h
#define tp_caffe2_utils_OpUtils_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{
struct ModelDetails;

//##################################################################################################
void removeOpByOutput(caffe2::NetDef& net,const std::string& opOutputName);

//##################################################################################################
void addGradientOps(ModelDetails& model);

//##################################################################################################
void addApplyGradientsOps_simple(ModelDetails& model, float lr);

//##################################################################################################
void addApplyGradientsOps_momentum(ModelDetails& model, float lr, float momentum);

//##################################################################################################
void addApplyGradientsOps_clippedMomentum(ModelDetails& model,
                                          float lr,
                                          float momentum,
                                          float minGradient,
                                          float maxGradient);

//##################################################################################################
void addApplyGradientsOps_adamOptimizer(ModelDetails& model,
                                        float initialLR,
                                        float initialIter,
                                        float beta1,
                                        float beta2,
                                        float epsilon);

}

#endif
