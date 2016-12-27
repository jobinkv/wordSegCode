#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "GCoptimization.h"
#include "LinkedBlockList.h"
#include <errno.h>
#include <list>
#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>

#define p_size 30
#define areaRem 250
#define feturE 1

using std::string;
using namespace cv;
using namespace std;

class layoutElement
{
   public:
      Rect box;   // rectangular cordinates
      int label;  // labels of the rectangle
      double cost; // the cost of the set of the layoutElement and it will attached to the first element
};


class tmpSearch {
  Mat unaryval;
  Mat isumText, isumGraphics, isumback;
  int imgHight,imgWidth;
  float lambda;
  vector<Rect> textboxes, graphicsBoxes; 
 public:
 tmpSearch(Mat);
 Rect initialBlock();
 Rect headingBlock(Rect bound, Rect bound2);
 void grapHBlock(Rect bound);
 void grapHBlock2(Rect bound); // 2 dimension
 Rect blockWithOut(Rect blk);
 Rect refineBlock(Rect block);
 vector<Rect> getGrahpBlock();
 vector<Rect> getTxtBlock();
  private: 
 double textEnrg(Rect);
 double graEnrg(Rect);
 double bagEnrg(Rect);
 Mat engyMin(int num_labels,Mat img,int lambda);
 

};
Mat GridGraph_Individually(int num_labels,Mat img,int lambda);
Mat rectPrior1(Mat layout);// independant width and height
Mat rectPrior4(Mat layout);//
Mat rectPrior2(Mat layout);//
Mat rectPrior3(Mat layout);// new approach
vector<Rect> OptXYcutH(Mat prbImg);//
vector<Rect> OptXYcutV(Mat prbImg);//
vector<Rect> detBlok(Mat binary);//
Mat OptXYcut(Mat prbImg);//

//constrained graph cut
Mat constrainedGC(Mat prbImg);//
Mat rectPrior20(Mat layout, Mat layout1, string temp_folder);

class tmpSearch2 {

 // vector<Rect> textboxes, graphicsBoxes; 
 public:
   Mat laySplit[3], Unaries[6];  
// Mat isumText, isumGraphics, isumback, dro, gra, nat, textHpadInt, textVpadInt;
 int imgHight,imgWidth;
 float lambda, gama;
 tmpSearch2(Mat, Mat);
 void clearmems();
 vector<layoutElement> energyCal(vector<layoutElement> layoutPrior);
 Mat CreatImgFmlayoutElementVec(vector<layoutElement> layoutPrior);
 vector<layoutElement> CreatlayoutElementVecFmImg(Mat image);
 //Rect initialBlock();
 //Rect headingBlock(Rect bound, Rect bound2);
 //void grapHBlock(Rect bound);
 //void grapHBlock2(Rect bound); // 2 dimension
 //Rect blockWithOut(Rect blk);
 //Rect refineBlock(Rect block);
 //vector<Rect> getGrahpBlock();
 //vector<Rect> getTxtBlock();
 //========================;
 //double energyCal(vector<layoutElement> layoutPrior);
 //
  private: // unari costs
 double unaryVal(Rect, int);
 
 //double graEnrg(Rect);
 //double bagEnrg(Rect);
 //double HoriCost(Rect); // calculate the cost feom horizontal
 //double VeriCost(Rect); // calculate the cost feom horizontal 
 //// pairwise cost (all foreground rectangles are isolated)
 //double textPairEnergy(Rect);
 //double grapPairEnergy(Rect);
 //Mat engyMin(int num_labels,Mat img,int lambda);
 };
 
 int getdir (string dir, vector<string> &files);
 void splitStr(const string& , char ,vector<string>& );
