#include <iostream>
#include "itkArray.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMedianImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkMultiScaleHessianEnhancementImageFilter.h"
#include "itkDescoteauxEigenToScalarImageFilter.h"


template <typename TImage>
void
Write(const TImage * out, std::string filename, bool compress)
{
  using WriterType = itk::ImageFileWriter<TImage>;
  typename WriterType::Pointer w = WriterType::New();
  w->SetInput(out);
  w->SetFileName(filename);
  w->SetUseCompression(compress);
  try
  {
    w->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << error << std::endl;
  }
}

template <typename TImage>
void
Write(itk::SmartPointer<TImage> out, std::string filename)
{
  Write(out.GetPointer(), filename.c_str());
}

template <typename TImage>
itk::SmartPointer<TImage>
connectedComponentAnalysis(const TImage * labelImage, std::string outFilename)
{
  using LabelImageType = itk::Image<itk::SizeValueType, TImage::ImageDimension>;
  using LabelerType = itk::ConnectedComponentImageFilter<TImage, LabelImageType>;
  LabelerType::Pointer labeler = LabelerType::New();
  labeler->SetInput(labelImage);
  static unsigned invocationCount = 0;
  Write(labeler->GetOutput(), outFilename + std::to_string(invocationCount) + "-cc-label.nrrd", true);

  using RelabelType = itk::RelabelComponentImageFilter<LabelImageType, TImage>;
  typename RelabelType::Pointer relabeler = RelabelType::New();
  relabeler->SetInput(labeler->GetOutput());
  relabeler->SetMinimumObjectSize(1000);
  Write(relabeler->GetOutput(), outFilename + std::to_string(invocationCount) + "-ccR-label.nrrd", true);
  ++invocationCount;

  return relabeler->GetOutput();
}

template <typename ImageType>
void
mainProcessing(typename ImageType::ConstPointer inImage, std::string outFilename, const itk::Array<double> & sigmaArray)
{
  constexpr unsigned ImageDimension = ImageType::ImageDimension;
  using OutImageType = itk::Image<unsigned char, ImageDimension>;
  using BinaryThresholdType = itk::BinaryThresholdImageFilter<ImageType, OutImageType>;

  typename BinaryThresholdType::Pointer binTh = BinaryThresholdType::New();
  binTh->SetInput(inImage);
  binTh->SetLowerThreshold(1000);
  Write(binTh->GetOutput(), outFilename + "-bin1-label.nrrd", true);

  typename OutImageType::Pointer thBone = connectedComponentAnalysis(binTh->GetOutput(), outFilename);

  using RealPixelType = float;
  using RealImageType = itk::Image<RealPixelType, ImageDimension>;
  // SDF

  using MultiScaleHessianFilterType = itk::MultiScaleHessianEnhancementImageFilter<ImageType, RealImageType>;
  using DescoteauxEigenToScalarImageFilterType =
    itk::DescoteauxEigenToScalarImageFilter<MultiScaleHessianFilterType::EigenValueImageType, RealImageType>;


  MultiScaleHessianFilterType::Pointer            multiScaleFilter = MultiScaleHessianFilterType::New();
  DescoteauxEigenToScalarImageFilterType::Pointer descoFilter = DescoteauxEigenToScalarImageFilterType::New();
  multiScaleFilter->SetInput(inImage);
  multiScaleFilter->SetEigenToScalarImageFilter(descoFilter);
  multiScaleFilter->SetSigmaArray(sigmaArray);

  // multiScaleFilter->Update();
  // Write(multiScaleFilter->GetOutput(), outFilename + "-desco.nrrd", false);
}

int
main(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0];
    std::cerr << " <InputFileName> <OutputSegmentation> [corticalBoneThickness]";
    std::cerr << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];
    double      corticalBoneThickness = 0.1;
    if (argc > 3)
    {
      corticalBoneThickness = std::stod(argv[3]);
    }

    constexpr unsigned nSigma = 5;
    itk::Array<double> sigmaArray;
    sigmaArray.SetSize(nSigma);
    for (int i = 0; i < nSigma; ++i)
    {
      sigmaArray.SetElement(i, (0.5 + i * 1.0 / (nSigma - 1)) * corticalBoneThickness);
    }
    std::cout << " InputFilePath: " << inputFileName << std::endl;
    std::cout << "OutputFilePath: " << outputFileName << std::endl;
    std::cout << "Sigmas: " << sigmaArray << std::endl;
    std::cout << std::endl;

    constexpr unsigned ImageDimension = 3;
    using InputPixelType = short;
    using InputImageType = itk::Image<InputPixelType, ImageDimension>;

    using ReaderType = itk::ImageFileReader<InputImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(inputFileName);

    using MedianType = itk::MedianImageFilter<InputImageType, InputImageType>;
    MedianType::Pointer median = MedianType::New();
    median->SetInput(reader->GetOutput());
    median->Update();

    InputImageType::Pointer inImage = median->GetOutput();
    inImage->DisconnectPipeline();

    mainProcessing<InputImageType>(inImage, outputFileName, sigmaArray);
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
