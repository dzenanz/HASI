/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkLandmarkAtlasSegmentationFilter_hxx
#define itkLandmarkAtlasSegmentationFilter_hxx

#include "itkLandmarkAtlasSegmentationFilter.h"

#include "itkLandmarkBasedTransformInitializer.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageRegistrationMethod.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkResampleImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"

#include "itkElastixRegistrationMethod.h"
#include "itkTransformixFilter.h"

#include "itkTransformFileWriter.h"
#include "itkImageFileWriter.h"

std::string outputBase = "./HASI";


namespace itk
{
template <typename TInputImage, typename TOutputImage>
void
LandmarkAtlasSegmentationFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TInputImage, typename TOutputImage>
template <typename TImage>
typename TImage::Pointer
LandmarkAtlasSegmentationFilter<TInputImage, TOutputImage>::Duplicate(const TImage * input, const RegionType subRegion)
{
  using DuplicatorType = itk::RegionOfInterestImageFilter<TImage, TImage>;
  typename DuplicatorType::Pointer roi = DuplicatorType::New();
  roi->SetInput(input);
  roi->SetRegionOfInterest(subRegion);
  roi->Update();
  return roi->GetOutput();
}

template <typename TInputImage, typename TOutputImage>
void
LandmarkAtlasSegmentationFilter<TInputImage, TOutputImage>::AffineFromRigid()
{
  m_AffineTransform = AffineTransformType::New();
  m_AffineTransform->SetCenter(m_RigidTransform->GetCenter());
  m_AffineTransform->SetTranslation(m_RigidTransform->GetTranslation());
  m_AffineTransform->SetMatrix(m_RigidTransform->GetMatrix());
  if (this->GetDebug())
  {
    WriteTransform(m_AffineTransform, outputBase + "-affineInit.tfm"); // debug
  }
}

template <typename TImage>
void
WriteImage(TImage * out, std::string filename, bool compress)
{
  using WriterType = itk::ImageFileWriter<TImage>;
  typename WriterType::Pointer w = WriterType::New();
  w->SetInput(out);
  w->SetFileName(filename);
  w->SetUseCompression(compress);
  w->Update();
}

void
WriteTransform(const itk::Object * transform, std::string fileName)
{
  using TransformWriterType = itk::TransformFileWriterTemplate<double>;
  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput(transform);
  transformWriter->SetFileName(fileName);
  transformWriter->Update();
}

template <typename TInputImage, typename TOutputImage>
void
LandmarkAtlasSegmentationFilter<TInputImage, TOutputImage>::GenerateData()
{
  this->AllocateOutputs();

  OutputImageType *      output = this->GetOutput();
  const InputImageType * input = this->GetInput();
  const RegionType &     outputRegion = output->GetRequestedRegion();
  RegionType             inputRegion = RegionType(outputRegion.GetSize());


  m_LandmarksTransform = RigidTransformType::New();

  //move this to VerifyPreconditions
  itkAssertOrThrowMacro(m_InputLandmarks.size() == 3, "There must be exactly 3 input landmarks");
  itkAssertOrThrowMacro(m_AtlasLandmarks.size() == 3, "There must be exactly 3 atlas landmarks");
  itkAssertOrThrowMacro(m_AtlasLabels.IsNotNull(), "AtlasLabels must be set");

  using LandmarkBasedTransformInitializerType =
    itk::LandmarkBasedTransformInitializer<RigidTransformType, InputImageType, InputImageType>;
  typename LandmarkBasedTransformInitializerType::Pointer landmarkBasedTransformInitializer =
    LandmarkBasedTransformInitializerType::New();

  landmarkBasedTransformInitializer->SetFixedLandmarks(m_InputLandmarks);
  landmarkBasedTransformInitializer->SetMovingLandmarks(m_AtlasLandmarks);

  m_LandmarksTransform->SetIdentity();
  landmarkBasedTransformInitializer->SetTransform(m_LandmarksTransform);
  landmarkBasedTransformInitializer->InitializeTransform();

  // force rotation to be around center of femur head
  m_LandmarksTransform->SetCenter(m_InputLandmarks.front());
  // and make sure that the other corresponding point maps to it perfectly
  m_LandmarksTransform->SetTranslation(m_AtlasLandmarks.front() - m_InputLandmarks.front());

  if (this->GetDebug())
  {
    WriteTransform(m_LandmarksTransform, outputBase + "-landmarks.tfm");
  }

  class ShowProgress : public itk::Command
  {
  public:
    itkNewMacro(ShowProgress);

    void
    Execute(itk::Object * caller, const itk::EventObject & event) override
    {
      Execute((const itk::Object *)caller, event);
    }

    void
    Execute(const itk::Object * caller, const itk::EventObject & event) override
    {
      if (!itk::ProgressEvent().CheckEvent(&event))
      {
        return;
      }
      const auto * processObject = dynamic_cast<const itk::ProcessObject *>(caller);
      if (!processObject)
      {
        return;
      }
      std::cout << " " << processObject->GetProgress();
    }
  };
  ShowProgress::Pointer showProgress = ShowProgress::New();

  InputImageType * inputBone1 = const_cast<InputImageType *>(this->GetInput(0));
  InputImageType * atlasBone1 = const_cast<InputImageType *>(this->GetInput(1));

  using ElastixParameterObject = elastix::ParameterObject;
  typename ElastixParameterObject::Pointer parameters = ElastixParameterObject::New();
  auto                                     rigid = parameters->GetDefaultParameterMap("rigid");
  parameters->SetParameterMap(rigid);

  using ElastixRMType = itk::ElastixRegistrationMethod<InputImageType, InputImageType>;
  typename ElastixRMType::Pointer elastixRM = ElastixRMType::New();

  typename ElastixRMType::ParameterObjectPointer elastixParameters = elastix::ParameterObject::New();
  elastixParameters->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("rigid"));
  elastixParameters->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("affine"));
  elastixParameters->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("bspline"));
  elastixParameters->SetParameter("FixedInternalImagePixelType", "short");
  elastixParameters->SetParameter("MovingInternalImagePixelType", "short");

  elastixRM->AddObserver(itk::ProgressEvent(), showProgress);
  elastixRM->SetParameterObject(elastixParameters);
  elastixRM->SetFixedImage(inputBone1);
  elastixRM->SetMovingImage(atlasBone1);
  elastixRM->SetOutputDirectory("M:/a/");
  elastixRM->SetLogFileName("elxLASf.log");
  elastixRM->SetLogToFile(true);
  //elastixRM->SetLogToConsole(false);

  elastixRM->Update();

  elastixRM->GetTransformParameterObject()->Print(std::cout);

  elastixParameters = elastixRM->GetTransformParameterObject();

  typename ElastixRMType::ParameterObjectPointer transformParameters = elastix::ParameterObject::New();
  transformParameters->SetParameterMap(elastixParameters->GetParameterMap(2));

  using TransformixFilterType = itk::TransformixFilter<TOutputImage>;
  TransformixFilterType::Pointer transformix = TransformixFilterType::New();
  transformix->SetMovingImage(m_AtlasLabels);
  transformix->SetTransformParameterObject(elastixParameters);

  //// grafting pattern spares us from allocating an intermediate image
  //transformix->GraftOutput(this->GetOutput());
  //transformix->Update();
  //this->GraftOutput(transformix->GetOutput());
  transformix->Update();
  if (this->GetDebug())
  {
    WriteImage(transformix->GetOutput(), outputBase + "-label.nrrd", true);
  }
  output = transformix->GetOutput();
}

} // end namespace itk

#endif // itkLandmarkAtlasSegmentationFilter_hxx
