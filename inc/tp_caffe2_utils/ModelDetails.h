#ifndef tp_caffe2_utils_ModelDetails_h
#define tp_caffe2_utils_ModelDetails_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

struct SubNetDetails
{
  caffe2::NetDef trainNet;
  std::vector<caffe2::OperatorDef*> gradientOps;
};

//##################################################################################################
struct ModelDetails
{
  caffe2::NetDef initPredictNet;
  caffe2::NetDef initTrainNet;

  caffe2::NetDef predictNet;
  caffe2::NetDef trainNet;

  std::vector<std::shared_ptr<SubNetDetails>> trainSubNets;

  caffe2::Workspace workspace;

  std::vector<caffe2::OperatorDef*> gradientOps;

  //! Ops to run only on the CPU.
  std::unordered_set<caffe2::OperatorDef*> cpuOps;

  //The names of blobs that are learnt as we train the network.
  std::vector<std::string> learntBlobNames;
};

}

#endif
