#ifndef tp_caffe2_utils_LearningRate_h
#define tp_caffe2_utils_LearningRate_h

#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
class AbstractLRPolicy
{
public:
  //################################################################################################
  virtual ~AbstractLRPolicy();

  //################################################################################################
  virtual void update(size_t iter, float loss);

  //################################################################################################
  size_t iter() const;

  //################################################################################################
  float loss() const;

  //################################################################################################
  float averageLoss() const;

  //################################################################################################
  float lr() const;

  //################################################################################################
  //! Returns a string describing the current state of the LR policy
  virtual std::string prettyStatus() const;

protected:
  //################################################################################################
  void updateBase(size_t iter, float loss);

  //################################################################################################
  void setIter(size_t iter);

  //################################################################################################
  void setLR(float lr);

private:
  size_t m_iter{0};
  float m_loss{0.0f};
  float m_averageLoss{0.0f};
  float m_lr{0.0f};
};

//##################################################################################################
class DecayOnPlateauLRPolicy: public AbstractLRPolicy
{
public:
  //################################################################################################
  DecayOnPlateauLRPolicy(float initialLR, float minLR);

  //################################################################################################
  void update(size_t iter, float loss) override;

  //################################################################################################
  std::string prettyStatus() const override;

private:
  float m_initialLR;
  float m_minLR;
};

//##################################################################################################
template<typename T, typename... Args>
class RampUpLRPolicy: public T
{
public:
  //################################################################################################
  RampUpLRPolicy(size_t rampUpIter, float rampUpFrom, Args... args):
    T(args...),
    m_rampUpIter(rampUpIter),
    m_rampUpFrom(rampUpFrom)
  {
    T::setLR(rampUpFrom);
  }

  //################################################################################################
  void update(size_t iter, float loss) override
  {
    if(iter>=m_rampUpIter)
    {
      T::update(iter-m_rampUpIter, loss);
      T::setIter(iter);
      return;
    }
    else if(iter==0)
    {
      T::update(0, loss);
      m_rampUpTo = T::lr();
    }

    T::updateBase(iter, loss);
    float f = float(iter) / float(m_rampUpIter);
    setLR((m_rampUpFrom*(1.0f-f))+(m_rampUpTo*f));
  }

  //################################################################################################
  std::string prettyStatus() const override
  {
    float f = tpMin(float(T::iter()) / float(m_rampUpIter), 1.0f);
    return "Ramp: " + std::to_string(f) + " " + T::prettyStatus();
  }

private:
  size_t m_rampUpIter;
  float m_rampUpFrom;
  float m_rampUpTo{0.0f};
};


}

#endif

