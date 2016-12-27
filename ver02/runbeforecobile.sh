#!/bin/bash
#rm *.xml
rm docSeg
#cmake ..
make
./docSeg


# book1 data....................
tranLoc="/mnt/data/datasets/bookDataSetNaccessories/TrainingAndTestingSetOfAllLang_cvpr/onlyOneBook/train"
gtLoc="/mnt/data/datasets/bookDataSetNaccessories/allGt"
testLoc="/mnt/data/datasets/bookDataSetNaccessories/TrainingAndTestingSetOfAllLang_cvpr/onlyOneBook/inputImg"
opLoc="/mnt/data/datasets/bookDataSetNaccessories/TrainingAndTestingSetOfAllLang_cvpr/onlyOneBook/gmmOut"
visOp="/mnt/data/datasets/bookDataSetNaccessories/TrainingAndTestingSetOfAllLang_cvpr/onlyOneBook/visvalOP"
#CVPR
CvprTrainImges="/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/newTrain"
CvprTrainGt="/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/newtrGt"
CvprTestImges="/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/testImg"
CvprTestGt="/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/ieeeTestGt"
CvprOutImages="/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/templatesOps"

cvpr2015Train="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/old100TrainingSet/orgimages"
cvpr2015TrainGt="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/old100TrainingSet/groundTimages"
cvpr2015Test="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/1-100"
cvpr2015TestGT="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/gt1-100"
cvpr2015Out="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/out1-100"
#sample try
#gt="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/sampletry/gt"
test="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/sampletry/org"
out="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/sampletry/out"
temps="/mnt/data/datasets/ieeePaper/FullCVPRPDF/tif_cvpr_2015/processed/forAnnotation/sampletry/gt"
#sample test
orgimg="/mnt/data/datasets/ieeePaper/cvprTemplates/orgimg"
out1="/mnt/data/datasets/ieeePaper/cvprTemplates/out"
template="/mnt/data/datasets/ieeePaper/cvprTemplates/template"
#./docSeg gmmtestFromSeparateModel $orgimg $out1 modelNw100.xml $template
# headding trends
inputimages="/mnt/data/datasets/ieeePaper/tif_cvpr_pageImageFull/Heading_pages"

#./docSeg gmmGbrTr $cvpr2015Train $cvpr2015TrainGt modelNw100.xml #modelNw = train30 images, modelNw100 = 100 images
#./docSeg gmmGrTestF $test $out modelNw100.xml #gmmtestFromSeparateModel
#rm result.txt
#./docSeg evalF $cvpr2015Out $cvpr2015TestGT result1000.csv
#cat result.txt
#crtTrainFetForFolder,TrainAgmmModelFrmFolder
#./docSeg visual $CvprTestImges $CvprOutImages $visOp
# folder training
blk="/mnt/data/datasets/ieeePaper/documentFigureClasses/MoreSrtuctured/Train/blk" # blockdiagram
dro="/mnt/data/datasets/ieeePaper/documentFigureClasses/MoreSrtuctured/Train/dro" # drowing
gra="/mnt/data/datasets/ieeePaper/documentFigureClasses/MoreSrtuctured/Train/gra" # graphics
nat="/mnt/data/datasets/ieeePaper/documentFigureClasses/MoreSrtuctured/Train/nat" # natural images
#./docSeg gmmGbrTrFromFolder $blk blk.xml
#./docSeg gmmGbrTrFromFolder $dro dro.xml
#./docSeg gmmGbrTrFromFolder $gra gra.xml
#./docSeg gmmGbrTrFromFolder $nat nat.xml

#new function




