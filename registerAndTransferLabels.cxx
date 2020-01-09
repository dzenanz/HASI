#include <chrono>
#include <iostream>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLandmarkBasedTransformInitializer.h"
#include "itkTransformFileWriter.h"

#include "itkImageRegistrationMethod.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkResampleImageFilter.h"
#include "itkCommand.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"
#include "itkSqrtImageFilter.h"

auto startTime = std::chrono::steady_clock::now();

template <typename TImage>
itk::SmartPointer<TImage>
ReadImage(std::string filename)
{
  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " Reading " << filename << std::endl;

  using ReaderType = itk::ImageFileReader<TImage>;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  reader->Update();
  itk::SmartPointer<TImage> out = reader->GetOutput();
  out->DisconnectPipeline();

  diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " Done!" << std::endl;
  return out;
}

template <typename TImage>
void
WriteImage(itk::SmartPointer<TImage> out, std::string filename, bool compress)
{
  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " Writing " << filename << std::endl;

  using WriterType = itk::ImageFileWriter<TImage>;
  typename WriterType::Pointer w = WriterType::New();
  w->SetInput(out);
  w->SetFileName(filename);
  w->SetUseCompression(compress);
  w->Update();

  diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " Done!" << std::endl;
}

void
WriteTransform(const itk::Object * transform, std::string fileName)
{
  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " Writing " << fileName << std::endl;

  using TransformWriterType = itk::TransformFileWriterTemplate<double>;
  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput(transform);
  transformWriter->SetFileName(fileName);
  transformWriter->Update();
}

std::vector<itk::Point<double, 3>>
readSlicerFiducials(std::string fileName)
{
  using PointType = itk::Point<double, 3>;
  std::ifstream pointsFile(fileName.c_str());
  std::string   line;
  // ignore first 3 lines (comments of fiducials savefile)
  std::getline(pointsFile, line); //# Markups fiducial file version = 4.10
  std::getline(pointsFile, line); //# CoordinateSystem = 0
  bool ras = false;
  if (line[line.length() - 1] == '0')
    ras = true;
  else if (line[line.length() - 1] == '1')
    ; // LPS, great
  else if (line[line.length() - 1] == '2')
    throw itk::ExceptionObject(
      __FILE__, __LINE__, "Fiducials file with IJK coordinates is not supported", __FUNCTION__);
  else
    throw itk::ExceptionObject(__FILE__, __LINE__, "Unrecognized coordinate system", __FUNCTION__);
  std::getline(pointsFile, line); //# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID

  std::vector<PointType> points;
  std::getline(pointsFile, line);
  while (!pointsFile.eof())
  {
    if (!pointsFile.good())
      break;
    PointType         p;
    std::stringstream iss(line);

    std::string val;
    std::getline(iss, val, ','); // ignore ID
    for (int col = 0; col < 3; ++col)
    {
      std::getline(iss, val, ',');
      if (!iss.good())
        break;

      std::stringstream convertor(val);
      convertor >> p[col];

      if (ras && col < 2)
        p[col] *= -1;
    }
    points.push_back(p);
    std::getline(pointsFile, line);
  }
  return points;
}

class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  CommandIterationUpdate() = default;

public:
  using OptimizerType = itk::RegularStepGradientDescentOptimizer;
  using OptimizerPointer = const OptimizerType *;

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    auto optimizer = static_cast<OptimizerPointer>(object);
    if (!(itk::IterationEvent().CheckEvent(&event)))
    {
      return;
    }
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
    std::cout << diff.count() << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << std::endl;
  }
};

template <typename ImageType>
void
mainProcessing(std::string inputBase, std::string outputBase, std::string atlasBase)
{
  constexpr unsigned Dimension = ImageType::ImageDimension;
  using LabelImageType = itk::Image<unsigned char, Dimension>;
  using RegionType = typename LabelImageType::RegionType;
  using IndexType = typename LabelImageType::IndexType;
  using SizeType = typename LabelImageType::SizeType;
  using PointType = typename ImageType::PointType;

  std::vector<PointType> inputLandmarks = readSlicerFiducials(inputBase + ".fcsv");
  std::vector<PointType> atlasLandmarks = readSlicerFiducials(atlasBase + ".fcsv");
  itkAssertOrThrowMacro(inputLandmarks.size() == 3, "There must be exactly 3 input landmarks");
  itkAssertOrThrowMacro(atlasLandmarks.size() == 3, "There must be exactly 3 atlas landmarks");

  using RigidTransformType = itk::VersorRigid3DTransform<double>;
  using LandmarkBasedTransformInitializerType =
    itk::LandmarkBasedTransformInitializer<RigidTransformType, ImageType, ImageType>;
  typename LandmarkBasedTransformInitializerType::Pointer landmarkBasedTransformInitializer =
    LandmarkBasedTransformInitializerType::New();

  landmarkBasedTransformInitializer->SetFixedLandmarks(inputLandmarks);
  landmarkBasedTransformInitializer->SetMovingLandmarks(atlasLandmarks);

  typename RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
  rigidTransform->SetIdentity();
  landmarkBasedTransformInitializer->SetTransform(rigidTransform);
  landmarkBasedTransformInitializer->InitializeTransform();

  // force rotation to be around center of femur head
  rigidTransform->SetCenter(inputLandmarks.front());
  // and make sure that the other corresponding point maps to it perfectly
  rigidTransform->SetTranslation(atlasLandmarks.front() - inputLandmarks.front());

  WriteTransform(rigidTransform, outputBase + "-landmarks.tfm");


  typename ImageType::Pointer inputBone1 = ReadImage<ImageType>(inputBase + "-bone1.nrrd");
  typename ImageType::Pointer atlasBone1 = ReadImage<ImageType>(atlasBase + "-bone1.nrrd");

  typename LabelImageType::Pointer inputLabels = ReadImage<LabelImageType>(inputBase + "-label.nrrd");
  typename LabelImageType::Pointer atlasLabels = ReadImage<LabelImageType>(atlasBase + "-label.nrrd");

  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " resampling the atlas into the space of input image" << std::endl;


  // now comes the registration part, first rigid (initialized by landmarks as computed above)
  // then affine and finaly deformable BSpline
  using AffineTransformType = itk::AffineTransform<double, Dimension>;
  constexpr unsigned int SplineOrder = 3;
  using CoordinateRepType = double;
  using DeformableTransformType = itk::BSplineTransform<CoordinateRepType, Dimension, SplineOrder>;
  using OptimizerType = itk::RegularStepGradientDescentOptimizer;
  using MetricType = itk::MeanSquaresImageToImageMetric<ImageType, ImageType>;
  using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
  using RegistrationType = itk::ImageRegistrationMethod<ImageType, ImageType>;

  typename MetricType::Pointer       metric = MetricType::New();
  typename OptimizerType::Pointer    optimizer = OptimizerType::New();
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  typename RegistrationType::Pointer registration = RegistrationType::New();

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetInterpolator(interpolator);
  registration->SetFixedImage(inputBone1);
  registration->SetMovingImage(atlasBone1);

  // Auxiliary identity transform.
  using IdentityTransformType = itk::IdentityTransform<double, Dimension>;
  IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();

  metric->ReinitializeSeed(76926294);

  ImageType::RegionType fixedRegion = inputBone1->GetBufferedRegion();
  registration->SetFixedImageRegion(fixedRegion);
  registration->SetInitialTransformParameters(rigidTransform->GetParameters());
  registration->SetTransform(rigidTransform);


  //  Define optimizer normalization to compensate for different dynamic range
  //  of rotations and translations.
  using OptimizerScalesType = OptimizerType::ScalesType;
  OptimizerScalesType optimizerScales(rigidTransform->GetNumberOfParameters());
  const double        translationScale = 1.0 / 1000.0;

  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;

  optimizer->SetScales(optimizerScales);

  optimizer->SetMaximumStepLength(0.2000);
  optimizer->SetMinimumStepLength(0.0001);

  optimizer->SetNumberOfIterations(200);

  // The rigid transform has 6 parameters we use therefore a few samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  metric->SetNumberOfSpatialSamples(10000L);
  metric->SetUseFixedImageSamplesIntensityThreshold(-1000);
  // metric->SetMovingImageMask(atlasLabels); // TODO: make this work

  // Create the Command observer and register it with the optimizer.
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);


  std::cout << "Starting Rigid Registration " << std::endl;
  registration->Update();

  std::cout << "Optimizer stop condition = " << registration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
  std::cout << "Rigid Registration completed" << std::endl;
  std::cout << std::endl;

  rigidTransform->SetParameters(registration->GetLastTransformParameters());
  WriteTransform(rigidTransform, outputBase + "-rigid.tfm");


  //  Perform Affine Registration
  AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  affineTransform->SetCenter(rigidTransform->GetCenter());
  affineTransform->SetTranslation(rigidTransform->GetTranslation());
  affineTransform->SetMatrix(rigidTransform->GetMatrix());

  registration->SetTransform(affineTransform);
  registration->SetInitialTransformParameters(affineTransform->GetParameters());

  optimizerScales = OptimizerScalesType(affineTransform->GetNumberOfParameters());
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  optimizerScales[4] = 1.0;
  optimizerScales[5] = 1.0;
  optimizerScales[6] = 1.0;
  optimizerScales[7] = 1.0;
  optimizerScales[8] = 1.0;

  optimizerScales[9] = translationScale;
  optimizerScales[10] = translationScale;
  optimizerScales[11] = translationScale;

  optimizer->SetScales(optimizerScales);
  optimizer->SetMaximumStepLength(0.2000);
  optimizer->SetMinimumStepLength(0.0001);
  optimizer->SetNumberOfIterations(200);

  // The Affine transform has 12 parameters we use therefore a more samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  metric->SetNumberOfSpatialSamples(50000L);

  std::cout << "Starting Affine Registration " << std::endl;
  registration->Update();

  std::cout << "Affine Registration completed" << std::endl;
  std::cout << std::endl;

  affineTransform->SetParameters(registration->GetLastTransformParameters());
  WriteTransform(affineTransform, outputBase + "-affine.tfm");


  //  Perform Deformable Registration
  typename DeformableTransformType::Pointer bsplineTransformCoarse = DeformableTransformType::New();

  unsigned int numberOfGridNodesInOneDimensionCoarse = 5;

  typename DeformableTransformType::PhysicalDimensionsType fixedPhysicalDimensions;
  typename DeformableTransformType::MeshSizeType           meshSize;
  typename DeformableTransformType::OriginType             fixedOrigin;

  for (unsigned int i = 0; i < Dimension; i++)
  {
    fixedOrigin[i] = inputBone1->GetOrigin()[i];
    fixedPhysicalDimensions[i] = inputBone1->GetSpacing()[i] * static_cast<double>(fixedRegion.GetSize()[i] - 1);
  }
  meshSize.Fill(numberOfGridNodesInOneDimensionCoarse - SplineOrder);

  bsplineTransformCoarse->SetTransformDomainOrigin(fixedOrigin);
  bsplineTransformCoarse->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
  bsplineTransformCoarse->SetTransformDomainMeshSize(meshSize);
  bsplineTransformCoarse->SetTransformDomainDirection(inputBone1->GetDirection());

  using ParametersType = DeformableTransformType::ParametersType;

  unsigned int numberOfBSplineParameters = bsplineTransformCoarse->GetNumberOfParameters();


  optimizerScales = OptimizerScalesType(numberOfBSplineParameters);
  optimizerScales.Fill(1.0);
  optimizer->SetScales(optimizerScales);


  ParametersType initialDeformableTransformParameters(numberOfBSplineParameters);
  initialDeformableTransformParameters.Fill(0.0);
  bsplineTransformCoarse->SetParameters(initialDeformableTransformParameters);
  registration->SetInitialTransformParameters(bsplineTransformCoarse->GetParameters());
  registration->SetTransform(bsplineTransformCoarse);

  optimizer->SetMaximumStepLength(10.0);
  optimizer->SetMinimumStepLength(0.01);
  optimizer->SetRelaxationFactor(0.7);
  optimizer->SetNumberOfIterations(50);

  // The BSpline transform has a large number of parameters, we use therefore a
  // much larger number of samples to run this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  metric->SetNumberOfSpatialSamples(numberOfBSplineParameters * 100);

  std::cout << std::endl << "Starting Deformable Registration Coarse Grid" << std::endl;
  registration->Update();

  std::cout << "Deformable Registration Coarse Grid completed" << std::endl;
  std::cout << std::endl;

  OptimizerType::ParametersType finalParameters = registration->GetLastTransformParameters();
  bsplineTransformCoarse->SetParameters(finalParameters);
  WriteTransform(bsplineTransformCoarse, outputBase + "-BSplineCoarse.tfm");


  DeformableTransformType::Pointer bsplineTransformFine = DeformableTransformType::New();
  unsigned int                     numberOfGridNodesInOneDimensionFine = 20;
  meshSize.Fill(numberOfGridNodesInOneDimensionFine - SplineOrder);

  bsplineTransformFine->SetTransformDomainOrigin(fixedOrigin);
  bsplineTransformFine->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
  bsplineTransformFine->SetTransformDomainMeshSize(meshSize);
  bsplineTransformFine->SetTransformDomainDirection(inputBone1->GetDirection());

  numberOfBSplineParameters = bsplineTransformFine->GetNumberOfParameters();

  ParametersType parametersHigh(numberOfBSplineParameters);
  parametersHigh.Fill(0.0);

  //  Now we need to initialize the BSpline coefficients of the higher resolution
  //  transform. This is done by first computing the actual deformation field
  //  at the higher resolution from the lower resolution BSpline coefficients.
  //  Then a BSpline decomposition is done to obtain the BSpline coefficient of
  //  the higher resolution transform.
  unsigned int counter = 0;
  for (unsigned int k = 0; k < Dimension; k++)
  {
    using ParametersImageType = DeformableTransformType::ImageType;
    using ResamplerType = itk::ResampleImageFilter<ParametersImageType, ParametersImageType>;
    ResamplerType::Pointer upsampler = ResamplerType::New();

    using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType, double>;
    FunctionType::Pointer function = FunctionType::New();

    upsampler->SetInput(bsplineTransformCoarse->GetCoefficientImages()[k]);
    upsampler->SetInterpolator(function);
    upsampler->SetTransform(identityTransform);
    upsampler->SetSize(bsplineTransformFine->GetCoefficientImages()[k]->GetLargestPossibleRegion().GetSize());
    upsampler->SetOutputSpacing(bsplineTransformFine->GetCoefficientImages()[k]->GetSpacing());
    upsampler->SetOutputOrigin(bsplineTransformFine->GetCoefficientImages()[k]->GetOrigin());

    using DecompositionType = itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType>;
    DecompositionType::Pointer decomposition = DecompositionType::New();

    decomposition->SetSplineOrder(SplineOrder);
    decomposition->SetInput(upsampler->GetOutput());
    decomposition->Update();

    ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();

    // copy the coefficients into the parameter array
    using Iterator = itk::ImageRegionIterator<ParametersImageType>;
    Iterator it(newCoefficients, bsplineTransformFine->GetCoefficientImages()[k]->GetLargestPossibleRegion());
    while (!it.IsAtEnd())
    {
      parametersHigh[counter++] = it.Get();
      ++it;
    }
  }

  optimizerScales = OptimizerScalesType(numberOfBSplineParameters);
  optimizerScales.Fill(1.0);
  optimizer->SetScales(optimizerScales);
  bsplineTransformFine->SetParameters(parametersHigh);

  //  We now pass the parameters of the high resolution transform as the initial
  //  parameters to be used in a second stage of the registration process.
  std::cout << "Starting Registration with high resolution transform" << std::endl;
  registration->SetInitialTransformParameters(bsplineTransformFine->GetParameters());
  registration->SetTransform(bsplineTransformFine);

  // The BSpline transform at fine scale has a very large number of parameters,
  // we use therefore a much larger number of samples to run this stage. In
  // this case, however, the number of transform parameters is closer to the
  // number of pixels in the image. Therefore we use the geometric mean of the
  // two numbers to ensure that the number of samples is larger than the number
  // of transform parameters and smaller than the number of samples.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  const unsigned int numberOfPixels = fixedRegion.GetNumberOfPixels();
  const auto         numberOfSamples = static_cast<unsigned long>(
    std::sqrt(static_cast<double>(numberOfBSplineParameters) * static_cast<double>(numberOfPixels)));
  metric->SetNumberOfSpatialSamples(numberOfSamples);
  registration->Update();

  std::cout << "Deformable Registration Fine Grid completed" << std::endl;
  std::cout << std::endl;

  finalParameters = registration->GetLastTransformParameters();
  bsplineTransformFine->SetParameters(finalParameters);
  WriteTransform(bsplineTransformFine, outputBase + "-BSplineFine.tfm");


  // Now resample the atlas labels into the input image space
  using ResampleFilterType = itk::ResampleImageFilter<LabelImageType, LabelImageType, double>;
  ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
  resampleFilter->SetInput(atlasLabels);
  resampleFilter->SetTransform(bsplineTransformFine);
  resampleFilter->SetReferenceImage(inputBone1);
  resampleFilter->SetUseReferenceImage(true);
  resampleFilter->SetDefaultPixelValue(0);
  resampleFilter->Update();
  typename LabelImageType::Pointer segmentedImage = resampleFilter->GetOutput();
  WriteImage(segmentedImage, outputBase + "-A-label.nrrd", true);


  using DifferenceFilterType = itk::SquaredDifferenceImageFilter<LabelImageType, LabelImageType, ImageType>;
  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();
  difference->SetInput1(inputLabels);
  difference->SetInput2(segmentedImage);
  using SqrtFilterType = itk::SqrtImageFilter<ImageType, LabelImageType>;
  SqrtFilterType::Pointer sqrtFilter = SqrtFilterType::New();
  sqrtFilter->SetInput(difference->GetOutput());
  sqrtFilter->Update();
  typename LabelImageType::Pointer diffImage = sqrtFilter->GetOutput();
  WriteImage(diffImage, outputBase + "-diff-label.nrrd", true);
}

int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage:\n" << argv[0];
    std::cerr << " <InputBase> <OutputBase> <AtlasBase>" << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    mainProcessing<itk::Image<short, 3>>(argv[1], argv[2], argv[3]);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject & exc)
  {
    std::cerr << exc;
  }
  catch (std::runtime_error & exc)
  {
    std::cerr << exc.what();
  }
  catch (...)
  {
    std::cerr << "Unknown error has occurred" << std::endl;
  }
  return EXIT_FAILURE;
}
