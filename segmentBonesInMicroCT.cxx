#include <iostream>
#include "itkArray.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMedianImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkNotImageFilter.h"
#include "itkMultiScaleHessianEnhancementImageFilter.h"
#include "itkDescoteauxEigenToScalarImageFilter.h"


template <typename TImage>
void
WriteImage(const TImage * out, std::string filename, bool compress)
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
WriteImage(itk::SmartPointer<TImage> out, std::string filename)
{
  WriteImage(out.GetPointer(), filename.c_str());
}

// split the binary mask into components and remove the small islands
template <typename TImage>
itk::SmartPointer<TImage>
connectedComponentAnalysis(const TImage * labelImage, std::string outFilename, itk::IdentifierType & numLabels)
{
  using ManyLabelImageType = itk::Image<itk::SizeValueType, TImage::ImageDimension>;
  using LabelerType = itk::ConnectedComponentImageFilter<TImage, ManyLabelImageType>;
  LabelerType::Pointer labeler = LabelerType::New();
  labeler->SetInput(labelImage);
  static unsigned invocationCount = 0;
  WriteImage(labeler->GetOutput(), outFilename + std::to_string(invocationCount) + "-cc-label.nrrd", true);

  using RelabelType = itk::RelabelComponentImageFilter<ManyLabelImageType, TImage>;
  typename RelabelType::Pointer relabeler = RelabelType::New();
  relabeler->SetInput(labeler->GetOutput());
  relabeler->SetMinimumObjectSize(1000);
  WriteImage(relabeler->GetOutput(), outFilename + std::to_string(invocationCount) + "-ccR-label.nrrd", true);
  ++invocationCount;

  relabeler->Update();
  numLabels = relabeler->GetNumberOfObjects();
  return relabeler->GetOutput();
}


// morphological dilation by thresholding the distance field
template <typename TImage>
itk::SmartPointer<TImage>
sdfDilate(itk::SmartPointer<TImage> labelImage, double radius, std::string outFilename)
{
  using RealPixelType = float;
  using RealImageType = itk::Image<RealPixelType, TImage::ImageDimension>;
  using DistanceFieldType = itk::SignedMaurerDistanceMapImageFilter<TImage, RealImageType>;

  typename DistanceFieldType::Pointer distF = DistanceFieldType::New();
  distF->SetInput(labelImage);
  distF->SetSquaredDistance(true);
  static unsigned invocationCount = 0;
  WriteImage(distF->GetOutput(), outFilename + std::to_string(invocationCount) + "-dist-dilate.nrrd", false);

  using FloatThresholdType = itk::BinaryThresholdImageFilter<RealImageType, TImage>;
  typename FloatThresholdType::Pointer sdfTh = FloatThresholdType::New();
  sdfTh->SetInput(distF->GetOutput());
  sdfTh->SetUpperThreshold(radius * radius);
  WriteImage(sdfTh->GetOutput(), outFilename + std::to_string(invocationCount) + "-dilate-label.nrrd", true);
  ++invocationCount;

  sdfTh->Update();
  return sdfTh->GetOutput();
}

// morphological erosion by thresholding the distance field
template <typename TImage>
itk::SmartPointer<TImage>
sdfErode(itk::SmartPointer<TImage> labelImage, double radius, std::string outFilename)
{
  // we need an inversion filter because Maurer's filter distances are not symmetrical
  // inside distances start at 0, while outside distances start at single spacing
  using NotType = itk::NotImageFilter<TImage, TImage>;
  NotType::Pointer negator = NotType::New();
  negator->SetInput(labelImage);
  static unsigned invocationCount = 0;
  WriteImage(negator->GetOutput(), outFilename + std::to_string(invocationCount) + "-erode-Not-label.nrrd", true);

  using RealPixelType = float;
  using RealImageType = itk::Image<RealPixelType, TImage::ImageDimension>;
  using DistanceFieldType = itk::SignedMaurerDistanceMapImageFilter<TImage, RealImageType>;

  typename DistanceFieldType::Pointer distF = DistanceFieldType::New();
  distF->SetInput(negator->GetOutput());
  distF->SetSquaredDistance(true);
  WriteImage(distF->GetOutput(), outFilename + std::to_string(invocationCount) + "-dist-erode.nrrd", false);

  using FloatThresholdType = itk::BinaryThresholdImageFilter<RealImageType, TImage>;
  typename FloatThresholdType::Pointer sdfTh = FloatThresholdType::New();
  sdfTh->SetInput(distF->GetOutput());
  sdfTh->SetLowerThreshold(radius * radius);
  WriteImage(sdfTh->GetOutput(), outFilename + std::to_string(invocationCount) + "-erode-label.nrrd", true);
  ++invocationCount;

  sdfTh->Update();
  return sdfTh->GetOutput();
}

template <typename ImageType>
void
mainProcessing(typename ImageType::ConstPointer inImage, std::string outFilename, const itk::Array<double> & sigmaArray)
{
  constexpr unsigned ImageDimension = ImageType::ImageDimension;
  using LabelImageType = itk::Image<unsigned char, ImageDimension>;
  using BinaryThresholdType = itk::BinaryThresholdImageFilter<ImageType, LabelImageType>;

  typename BinaryThresholdType::Pointer binTh = BinaryThresholdType::New();
  binTh->SetInput(inImage);
  binTh->SetLowerThreshold(1000);
  WriteImage(binTh->GetOutput(), outFilename + "-bin1-label.nrrd", true);

  itk::IdentifierType numBones = 0;

  typename LabelImageType::Pointer thBone = connectedComponentAnalysis(binTh->GetOutput(), outFilename, numBones);
  typename LabelImageType::Pointer dilatedBone =
    sdfDilate(thBone, 3.0 * sigmaArray[sigmaArray.size() - 1], outFilename);
  typename LabelImageType::Pointer erodedBone =
    sdfErode(dilatedBone, 3.5 * sigmaArray[sigmaArray.size() - 1], outFilename);


  using RealPixelType = float;
  using RealImageType = itk::Image<RealPixelType, ImageDimension>;

  using MultiScaleHessianFilterType = itk::MultiScaleHessianEnhancementImageFilter<ImageType, RealImageType>;
  using DescoteauxEigenToScalarImageFilterType =
    itk::DescoteauxEigenToScalarImageFilter<MultiScaleHessianFilterType::EigenValueImageType, RealImageType>;


  MultiScaleHessianFilterType::Pointer            multiScaleFilter = MultiScaleHessianFilterType::New();
  DescoteauxEigenToScalarImageFilterType::Pointer descoFilter = DescoteauxEigenToScalarImageFilterType::New();
  multiScaleFilter->SetInput(inImage);
  multiScaleFilter->SetEigenToScalarImageFilter(descoFilter);
  multiScaleFilter->SetSigmaArray(sigmaArray);

  // multiScaleFilter->Update();
  // WriteImage(multiScaleFilter->GetOutput(), outFilename + "-desco.nrrd", false);
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
