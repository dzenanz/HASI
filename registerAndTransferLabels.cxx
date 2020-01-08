#include <chrono>
#include <iostream>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLandmarkBasedTransformInitializer.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileWriter.h"


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

  using TransformType = itk::VersorRigid3DTransform<double>;
  using LandmarkBasedTransformInitializerType =
    itk::LandmarkBasedTransformInitializer<TransformType, ImageType, ImageType>;
  typename LandmarkBasedTransformInitializerType::Pointer landmarkBasedTransformInitializer =
    LandmarkBasedTransformInitializerType::New();

  //// we need more landmarks for AffineTransform which supports weights
  //// so duplicate the first (pivotal) point
  //inputLandmarks.push_back(inputLandmarks.front());
  //atlasLandmarks.push_back(atlasLandmarks.front());

  landmarkBasedTransformInitializer->SetFixedLandmarks(inputLandmarks);
  landmarkBasedTransformInitializer->SetMovingLandmarks(atlasLandmarks);

  // give the most weight to center of femur head, then to shaft, then to the dent
  typename LandmarkBasedTransformInitializerType::LandmarkWeightType weights{ 1e9, 1, 1e-9, 1e9 };
  landmarkBasedTransformInitializer->SetLandmarkWeight(weights);

  typename TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  landmarkBasedTransformInitializer->SetTransform(transform);
  landmarkBasedTransformInitializer->InitializeTransform();

  using TransformWriterType = itk::TransformFileWriterTemplate<double>;
  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput(transform);
  transformWriter->SetFileName(outputBase + ".tfm");
  transformWriter->Update();

  typename ImageType::Pointer inImage = ReadImage<ImageType>(inputBase + "-bone1.nrrd");
  //typename ImageType::Pointer atlasBone1 = ReadImage<ImageType>(atlasBase + "-bone1.nrrd");

  typename LabelImageType::Pointer atlasLabels = ReadImage<LabelImageType>(atlasBase + "-label.nrrd");

  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - startTime;
  std::cout << diff.count() << " resampling the atlas into the space of input image" << std::endl;

  using ResampleFilterType = itk::ResampleImageFilter<LabelImageType, LabelImageType, double>;
  ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
  resampleFilter->SetInput(atlasLabels);
  resampleFilter->SetTransform(transform);
  resampleFilter->SetReferenceImage(inImage);
  resampleFilter->SetUseReferenceImage(true);
  resampleFilter->SetDefaultPixelValue(-4096);
  resampleFilter->Update();

  typename LabelImageType::Pointer segmentedImage = resampleFilter->GetOutput();

  WriteImage(segmentedImage, outputBase + "-label.nrrd", true);
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
