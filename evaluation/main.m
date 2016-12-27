clc; clear all; close all;

outFolder='/users/jobinkv/2nd_data/word100img/validationOp/stage_6/';
grtFolder = '/users/jobinkv/2nd_data/word100img/gtTextFiles/';

   % output text files
   fileID = fopen('results_for6.txt','w');
   fprintf(fileID,'%19s\t %19s\t %19s\t %19s\t %19s\t %19s\n',...
       'image name','False Detection',' Missed Regions', 'Correct Segmentation',...
       'Over Segmentation', 'Under segmentation');

CurrectSeg=[];
FalseDetection=[];
MissedReg=[];
OverSeg=[];
UnderEsg=[];

outFiles=dir([outFolder,'*.txt']);
for i=1:length(outFiles)
   outFileNames=outFiles(i).name
   gtFileNames=[outFileNames(1:end-14),'.txt'];
   if (strcmp(gtFileNames,'1517_blk_6.txt'))
       continue;
   end
   outCordinate = dlmread([outFolder,outFileNames]);
   grtCordinate_row = dlmread([grtFolder,gtFileNames]);
   grtCordinate = grtCordinate_row(:,1:4);
   op=Evaluation_measures(grtCordinate,outCordinate);
   

   
   FalseDetection(i,1) = op(1)*100/size(outCordinate,1);
   MissedReg(i,1) = op(2)*100/size(grtCordinate,1);
   CurrectSeg(i,1) = op(8)*100/size(grtCordinate,1);
   OverSeg(i,1) = op(9)*100/size(grtCordinate,1);
   UnderEsg(i,1) = op(10)*100/size(grtCordinate,1);
   fprintf(fileID,'%19s\t %6.2f\t %6.2f\t %6.2f\t %6.2f\t %6.2f\n',...
   gtFileNames, FalseDetection(i,1), MissedReg(i,1), CurrectSeg(i,1), OverSeg(i,1), UnderEsg(i,1));

end
fprintf(fileID,'%19s\t %6.2f\t %6.2f\t %6.2f\t %6.2f\t %6.2f\n',...
   'Average', mean(FalseDetection), mean(MissedReg), mean(CurrectSeg), mean(OverSeg), mean(UnderEsg));


fclose(fileID);
