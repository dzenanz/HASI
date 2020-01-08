#include <chrono>
#include <iostream>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLandmarkBasedTransformInitializer.h"
#include "itkResampleImageFilter.h"


auto startTime = std::chrono::steady_clock::now();

template <typename TImage>
itk::SmartPointer<TImage>
ReadImage(std::string filename)
{
  auto diff = std::chrono::steady_clock::now() - startTime;
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
  auto diff = std::chrono::steady_clock::now() - startTime;
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
mainProcessing(std::string inputFileName, std::string outFilename, std::string atlasDirectory)
{
  constexpr unsigned Dimension = ImageType::ImageDimension;
  using LabelImageType = itk::Image<unsigned char, Dimension>;
  using RegionType = typename LabelImageType::RegionType;
  using IndexType = typename LabelImageType::IndexType;
  using SizeType = typename LabelImageType::SizeType;
  using PointType = typename ImageType::PointType;

  typename ImageType::Pointer inImage = ReadImage<ImageType>(inputFileName);
  typename ImageType::Pointer atlasBone1 = ReadImage<ImageType>(atlasDirectory + "/907-L-bone1.nrrd");
  std::vector<PointType> aLandmarks = readSlicerFiducials(atlasDirectory + "/907-L.fcsv");

  double avgSpacing = 1.0;
  for (unsigned d = 0; d < Dimension; d++)
  {
    avgSpacing *= inImage->GetSpacing()[d];
  }
  avgSpacing = std::pow(avgSpacing, 1.0 / Dimension); // geometric average preserves voxel volume
  float epsDist = 0.001 * avgSpacing;                 // epsilon for distance comparisons

  RegionType wholeImage = inImage->GetLargestPossibleRegion();

  typename ImageType::Pointer atlasLabels = ReadImage<ImageType>(atlasDirectory + "/907-L-label.nrrd");
  // resample into the space of inImage

  WriteImage(atlasLabels, outFilename, true);
}

int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0];
    std::cerr << " <InputFileName> <OutputFileName> <AtlasDirectory>";
    std::cerr << std::endl;
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
