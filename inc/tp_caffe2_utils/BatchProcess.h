#ifndef tp_caffe2_utils_BatchProcess_h
#define tp_caffe2_utils_BatchProcess_h

#include "tp_caffe2_utils/Globals.h"
#include "tp_caffe2_utils/ModelDetails.h"
#include "tp_caffe2_utils/NetUtils.h"

#include "tp_utils/MutexUtils.h"

#include <cassert>

namespace tp_caffe2_utils
{

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
  void initThreads(const std::vector<DeviceDetails>& devices)
  {
    assert(!devices.empty());

    for(const auto& device : devices)
    {
      m_totalBatchSize+=device.batchSize;
      m_threads.push_back(new std::thread([&, device]
      {
        ModelDetails model;
        prepareModel(device, model);

        tp_caffe2_utils::setDeviceType(model.initPredictNet, model.cpuOps, device.deviceType);
        tp_caffe2_utils::setDeviceType(model.predictNet    , model.cpuOps, device.deviceType);

        model.workspace.RunNetOnce(model.initPredictNet);
        model.workspace.CreateNet (model.predictNet);

        TP_MUTEX_LOCKER(m_mutex);
        for(;;)
        {
          auto run = [&]
          {
            if(m_items.empty())
              return false;

            if(!m_finish && m_items.size()<size_t(device.batchSize))
              return false;

            size_t i=tpMin(m_items.size(), size_t(device.batchSize));
            std::vector<std::shared_ptr<T>> items;
            items.resize(i);
            std::copy_n(m_items.begin(), i, items.begin());
            m_items.erase(m_items.begin(), m_items.begin()+i);

            m_waitCondition.wakeAll();

            {
              TP_MUTEX_UNLOCKER(m_mutex);
              feed(device, model, items);
              model.workspace.RunNet(model.predictNet.name());
              complete(device, model, items);
            }

            return true;
          };

          if(!run())
          {
            if(m_finish && m_items.empty())
              break;

            m_waitCondition.wait(TPMc m_mutex);
          }
        }
      }));
    }
  }

  //################################################################################################
  virtual void prepareModel(const DeviceDetails& device,
                            ModelDetails& model)=0;

  //################################################################################################
  virtual void feed(const DeviceDetails& device,
                    ModelDetails& model,
                    const std::vector<std::shared_ptr<T>>& items)=0;

  //################################################################################################
  virtual void complete(const DeviceDetails& device,
                        ModelDetails& model,
                        const std::vector<std::shared_ptr<T>>& items)=0;

  //################################################################################################
  void addItem(const std::shared_ptr<T>& data, size_t maxActive=std::numeric_limits<size_t>::max())
  {
    TP_MUTEX_LOCKER(m_mutex);

    while(m_items.size()>maxActive)
      m_waitCondition.wait(TPMc m_mutex);

    m_items.push_back(data);
    m_waitCondition.wakeAll();
  }

  //################################################################################################
  size_t totalBatchSize() const
  {
    return m_totalBatchSize;
  }

  //################################################################################################
  void join()
  {
    {
      TP_MUTEX_LOCKER(m_mutex);
      m_finish=true;
      m_waitCondition.wakeAll();
    }

    for(auto thread : m_threads)
    {
      thread->join();
      delete thread;
    }

    m_threads.clear();
  }

  //################################################################################################
  virtual ~BatchProcess()
  {
    join();
  }

private:
  TPMutex m_mutex{TPM};
  TPWaitCondition m_waitCondition;
  std::vector<std::thread*> m_threads;
  bool m_finish{false};
  std::vector<std::shared_ptr<T>> m_items;
  size_t m_totalBatchSize{0};
};

}

#endif

