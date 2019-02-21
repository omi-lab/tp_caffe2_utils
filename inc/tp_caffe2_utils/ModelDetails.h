#ifndef tp_caffe2_utils_ModelDetails_h
#define tp_caffe2_utils_ModelDetails_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
struct ModelDetails
{
  caffe2::NetDef initPredictNet;
  caffe2::NetDef initTrainNet;

  caffe2::NetDef predictNet;
  caffe2::NetDef trainNet;

  caffe2::Workspace workspace;

  std::vector<caffe2::OperatorDef*> gradientOps;

  //The names of blobs that are learnt as we train the network.
  std::vector<std::string> learntBlobNames;

  //The names of blobs that contain data as it passes forward and back through the network.
  std::vector<std::string> dataBlobNames;
};

}

#endif
