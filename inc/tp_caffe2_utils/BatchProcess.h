#ifndef tp_caffe2_utils_BatchProcess_h
#define tp_caffe2_utils_BatchProcess_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{
struct ModelDetails;

//##################################################################################################
struct DeviceDetails
{
  int32_t deviceType{defaultDeviceType};
  int32_t deviceID{-1};
  int64_t batchSize{1};
};

//##################################################################################################
template<typename T>
class BatchProcess
{
public:
  //################################################################################################
  BatchProcess(const std::vector<DeviceDetails>& devices);

  //################################################################################################
  virtual void feed(ModelDetails& model, const std::vector<T*>& items)=0;

  //################################################################################################
  virtual void complete(ModelDetails& model, const std::vector<T*>& items)=0;

  //################################################################################################
  void addItem(T* data)
  {

  }

  //################################################################################################
  void flush()
  {

  }

  //################################################################################################
  virtual ~BatchProcess()
  {

  }

private:
  std::vector<T*> m_items;
};

}

#endif

