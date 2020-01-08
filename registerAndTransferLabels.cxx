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


template <typename ImageType>
void
mainProcessing(std::string inputFileName, std::string outFilename, std::string atlasDirectory)
{
  constexpr unsigned Dimension = ImageType::ImageDimension;
  using LabelImageType = itk::Image<unsigned char, Dimension>;
  using RegionType = typename LabelImageType::RegionType;
  using IndexType = typename LabelImageType::IndexType;
  using SizeType = typename LabelImageType::SizeType;

  typename ImageType::Pointer inImage = ReadImage<ImageType>(inputFileName);
  typename ImageType::Pointer atlasBone1 = ReadImage<ImageType>(atlasDirectory + "/907-L-bone1.nrrd");

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
