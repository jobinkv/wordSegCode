#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include "GCoptimization.h"
#include "LinkedBlockList.h"
#include <errno.h>
#include <list>
#include "rectprior.h"
#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>



using std::string;
using namespace cv;
using namespace std;



tmpSearch::tmpSearch(Mat layout) 
{
	unaryval = layout;
	Mat laySplit[3];
	split(layout, laySplit);
	integral(laySplit[0],isumText,CV_64F);
	integral(laySplit[1],isumGraphics,CV_64F);
	integral(laySplit[2],isumback,CV_64F);	
	imgHight=layout.rows,imgWidth = layout.cols;
	lambda = .3;
	
}

double tmpSearch::textEnrg(Rect bound)
{
	double tl = isumText.at<double>(bound.y,bound.x);
	double tr = isumText.at<double>(bound.y,bound.x + bound.width);	
	double bl = isumText.at<double>(bound.y + bound.height,bound.x);
	double br = isumText.at<double>(bound.y + bound.height,bound.x + bound.width);
	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
		return  br-bl-tr+tl;
	else
		return 0;
}

double tmpSearch::graEnrg(Rect bound)
{
	double tl = isumGraphics.at<double>(bound.y,bound.x);
	double tr = isumGraphics.at<double>(bound.y,bound.x + bound.width);	
	double bl = isumGraphics.at<double>(bound.y + bound.height,bound.x);
	double br = isumGraphics.at<double>(bound.y + bound.height,bound.x + bound.width);
	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
		return  br-bl-tr+tl;
	else
		return 0;
}

double tmpSearch::bagEnrg(Rect bound)
{
	double tl = isumback.at<double>(bound.y,bound.x);
	double tr = isumback.at<double>(bound.y,bound.x + bound.width);	
	double bl = isumback.at<double>(bound.y + bound.height,bound.x);
	double br = isumback.at<double>(bound.y + bound.height,bound.x + bound.width);
	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
		return  br-bl-tr+tl;
	else
		return 0;
}

Rect tmpSearch::initialBlock()
{

	double minScoreT = 1e10, textUnar=0, graUnar=0, bacUnar=0, bacUnarbntr=0, graUnarbntr=0, textUnarbntr=0,unaryScoreTnG1 = 0, pairwiseScore1 = 0, templtCost1=0; // large value bntr
	Rect bound;

	//cout<<"expected cetroid ="<<cetroidExp<<endl;
	int stepSize=p_size, stride=5;

	for (int i=stride;i<imgHight-stride-1;i+=stepSize) // top 
	for (int j=imgHight-stride-1;j>stride;j-=stepSize) // bottom 
	for (int k=stride;k<imgWidth*1-stride-1;k+=stepSize) // left
	for (int l=imgWidth*1-stride-1;l>stride;l-=stepSize) // right
	if (i<j and k<l and 2*(l-k)<imgWidth)
	{
		int wid=l-k,high = j-i;

		float ppc = wid*high/(2*wid*stride+2*high*stride+4*stride*stride);
		
		Rect valBox; // initial block
		valBox.x = k;
		valBox.y = i;
		valBox.width = wid;
		valBox.height = high;
		
		textUnar = textEnrg(valBox);
		graUnar = graEnrg(valBox);
		bacUnar = bagEnrg(valBox);
		
		valBox += Size(2*stride, 2*stride);
		valBox -= Point(stride,stride);
		
		
		bacUnarbntr = bagEnrg(valBox) - bacUnar;
		graUnarbntr = graEnrg(valBox) - graUnar;
		textUnarbntr= textEnrg(valBox) - textUnar;	
		


		unaryScoreTnG1 = 2*bacUnar -textUnar - graUnar;

		pairwiseScore1 = ppc*(textUnarbntr+graUnarbntr - 2*bacUnarbntr);	
		templtCost1 = (1-lambda)*unaryScoreTnG1 + lambda*pairwiseScore1;	
	

		if (templtCost1<minScoreT)
		{
			minScoreT=templtCost1;
			bound = valBox;
					
		}
		
	}	

	return bound;
}


Rect tmpSearch::blockWithOut(Rect blk)
{

	double minScoreT = 1e10, textUnar=0, graUnar=0, bacUnar=0, bacUnarbntr=0, graUnarbntr=0, textUnarbntr=0,unaryScoreTnG1 = 0, pairwiseScore1 = 0, templtCost1=0; // large value bntr
	Rect bound;

	//cout<<"expected cetroid ="<<cetroidExp<<endl;
	int stepSize=p_size, stride=5;

	for (int i=stride;i<imgHight-stride-1;i+=stepSize) // top 
	for (int j=imgHight-stride-1;j>stride;j-=stepSize) // bottom 
	for (int k=stride;k<imgWidth*1-stride-1;k+=stepSize) // left
	for (int l=imgWidth*1-stride-1;l>stride;l-=stepSize) // right
	if (i<j and k<l and 2*(l-k)<imgWidth)
	{
		int wid=l-k,high = j-i;
		Rect valBox; // initial block
		valBox.x = k;
		valBox.y = i;
		valBox.width = wid;
		valBox.height = high;
		
		Rect inter = valBox & blk;
		
		if (inter.width == 0 and inter.height == 0)
		{	
		

		float ppc = wid*high/(2*wid*stride+2*high*stride+4*stride*stride);
		

		
		textUnar = textEnrg(valBox);
		graUnar = graEnrg(valBox);
		bacUnar = bagEnrg(valBox);
		
		valBox += Size(2*stride, 2*stride);
		valBox -= Point(stride,stride);
		
		
		bacUnarbntr = bagEnrg(valBox) - bacUnar;
		graUnarbntr = graEnrg(valBox) - graUnar;
		textUnarbntr= textEnrg(valBox) - textUnar;	
		


		unaryScoreTnG1 = 2*bacUnar -textUnar - graUnar;

		pairwiseScore1 = ppc*(textUnarbntr+graUnarbntr - 2*bacUnarbntr);	
		templtCost1 = (1-lambda)*unaryScoreTnG1 + lambda*pairwiseScore1;	
	

		if (templtCost1<minScoreT)
		{
			minScoreT=templtCost1;
			bound = valBox;
					
		}
		
	}	
	}
	return bound;
}

Rect tmpSearch::headingBlock(Rect bound, Rect bound2)
{

	double minScoreT = 1e10, textUnar=0, graUnar=0, bacUnar=0, bacUnarbntr=0, graUnarbntr=0, textUnarbntr=0,unaryScoreTnG1 = 0, pairwiseScore1 = 0, templtCost1=0; // large value bntr
	Rect boundout;

	//cout<<"expected cetroid ="<<cetroidExp<<endl;
	int stepSize=p_size, stride=5;

	for (int i=stride;i<imgHight*1-stride-1;i+=stepSize) // top 
	for (int j=imgHight*1-stride-1;j>stride;j-=stepSize) // bottom 
	for (int k=stride;k<imgWidth*1-stride-1;k+=stepSize) // left
	for (int l=imgWidth*1-stride-1;l>stride;l-=stepSize) // right
	if (i<j and k<l )//and (l-k)>(j-i)
	{
		int wid=l-k,high = j-i;

		float ppc = wid*high/(2*wid*stride+2*high*stride+4*stride*stride);
		
		Rect valBox; // initial block
		valBox.x = k;
		valBox.y = i;
		valBox.width = wid;
		valBox.height = high;
		
		Rect inter = valBox & bound, inter2 = valBox & bound2;
		if (inter.width == 0 and inter2.height == 0)
		{
			textUnar = textEnrg(valBox);
			graUnar = graEnrg(valBox);
			bacUnar = bagEnrg(valBox);
		
			valBox += Size(2*stride, 2*stride);
			valBox -= Point(stride,stride);
		
		
			bacUnarbntr = bagEnrg(valBox) - bacUnar;
			graUnarbntr = graEnrg(valBox) - graUnar;
			textUnarbntr= textEnrg(valBox) - textUnar;	
		


			unaryScoreTnG1 = .5*bacUnar -textUnar - graUnar;

			pairwiseScore1 = ppc*(textUnarbntr+graUnarbntr - .5*bacUnarbntr);	
			templtCost1 = (1-lambda)*unaryScoreTnG1 + lambda*pairwiseScore1;	
	

			if (templtCost1<minScoreT)
			{
				minScoreT=templtCost1;
				boundout = valBox;
			}
		}
		
	}	

	return boundout;
}


Rect tmpSearch::refineBlock(Rect valBox)
{

	double minScoreT = 1e10, textUnar=0, graUnar=0, bacUnar=0, bacUnarbntr=0, graUnarbntr=0, textUnarbntr=0,unaryScoreTnG1 = 0, pairwiseScore1 = 0, templtCost1=0; // large value bntr
	
	Rect bound;
	int stepSize=p_size, stride=5;

	for (int i=-50;i<50;i+=2) // top 
	for (int j=-50;j<50;j+=2) // bottom 
	for (int k=-50;k<50;k+=2) // left
	for (int l=-50;l<50;l+=2) // right
	{
		
		valBox += Size(2*i,2*j);
		valBox -= Point(k,l);

		Rect bountBox;
		bountBox = valBox;
		
		bountBox += Size(2*stride, 2*stride);
		bountBox -= Point(stride,stride);
		
		if (bountBox.x >0 and bountBox.y>0 and bountBox.width>0 and bountBox.height>0)
		{
			float ppc = valBox.width*valBox.height/(2*valBox.width*stride+2*valBox.height*stride+4*stride*stride);
		
			textUnar = textEnrg(valBox);
			graUnar = graEnrg(valBox);
			bacUnar = bagEnrg(valBox);

			bacUnarbntr = bagEnrg(bountBox) - bacUnar;
			graUnarbntr = graEnrg(bountBox) - graUnar;
			textUnarbntr= textEnrg(bountBox) - textUnar;	
		
			unaryScoreTnG1 = 2*bacUnar -textUnar - graUnar;
			pairwiseScore1 = ppc*(textUnarbntr+graUnarbntr - 2*bacUnarbntr);	
			
			templtCost1 = (1-lambda)*unaryScoreTnG1 + lambda*pairwiseScore1;	
	

			if (templtCost1<minScoreT)
			{
				minScoreT=templtCost1;
				bound = valBox;
					
			}
		
		}	
	}
	return bound;
}

Mat tmpSearch::engyMin(int num_labels,Mat img,int lambda)
{

	int height=img.rows;//HEIGHT
	int width=img.cols;//width
	int num_pixels=height*width;

	int *result = new int[num_pixels];   // stores result of optimization
	int rw;
	int col;
	Mat  opimage =img.clone();
//image is transformed int 1 drow in row major order

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually


		for ( int i = 0; i < num_pixels; i++ )
		{
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;

			}	
			else
			{
			rw=(i+1)/width;
			col=((i+1)%width)-1;
			}

			int blue=img.at<cv::Vec3b>(rw,col)[0];
			int green=img.at<cv::Vec3b>(rw,col)[1];
			int red=img.at<cv::Vec3b>(rw,col)[2];



			for (int l = 0; l < num_labels; l++ )
			{
				if(l==0)
					 gc->setDataCost(i,l,(255-blue)/*+red+green*/);
			 	if(l==1)
			 		gc->setDataCost(i,l,(255-green)/*+red+blue*/);
		 		if(l==2)
		 			gc->setDataCost(i,l,(255-red)/*+blue+green*/);

			}
		}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{

				if(l1==l2)
				//int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
				gc->setSmoothCost(l1,l2,0);

				else

				gc->setSmoothCost(l1,l2,lambda);


			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());


		

		for ( int  i = 0; i < num_pixels; i++ )
		{
			result[i] = gc->whatLabel(i);
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;
			}
			else
			{
				rw=(i+1)/width;
				col=((i+1)%width)-1;
			}
			if(result[i]==0) //sky
			{
		//cout<<"label 0 \n";
				opimage.at<cv::Vec3b>(rw,col)[0]=255;//blue
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=0;
			}
			if(result[i]==1) // grass
			{
			opimage.at<cv::Vec3b>(rw,col)[0]=0;
			opimage.at<cv::Vec3b>(rw,col)[1]=255;
			opimage.at<cv::Vec3b>(rw,col)[2]=0;
			//cout<<"label 1 \n";
			}
			if(result[i]==2) //third object
			{
				opimage.at<cv::Vec3b>(rw,col)[0]=0;
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=255;//red
			}
		}


		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}
	delete [] result;
	return opimage;
}


void tmpSearch::grapHBlock(Rect bound)
{

	Mat unaryVal(Size(1,bound.height+2), CV_8UC3,Scalar(0,0,0)); 
// down sampling ///////// 1

	for (int i=1;i<bound.height+1;i+=1) // top 
	{
		Rect valBox; // initial block
		valBox.x = bound.x;
		valBox.y = bound.y + i;
		valBox.width = bound.width;
		valBox.height = 1;

		unaryVal.at<Vec3b>(0,i)[0] = textEnrg(valBox)/bound.width ;
		unaryVal.at<Vec3b>(0,i)[1] = (graEnrg(valBox)/bound.width+ bagEnrg(valBox)/bound.width)/2;
		unaryVal.at<Vec3b>(0,i)[2] = bagEnrg(valBox)/bound.width;

	}
	
// down sampling 2/////////////////
//	Mat imgtmp = unaryval(bound);
//	resize( imgtmp, unaryVal, Size( 1, imgtmp.rows ) );

		
	Mat enerfyMin = engyMin(2,unaryVal,100);
	enerfyMin.at<Vec3b>(0,0)[0] = 0;
	enerfyMin.at<Vec3b>(0,0)[1] = 0;
	enerfyMin.at<Vec3b>(0,0)[2] = 0;

	enerfyMin.at<Vec3b>(0,bound.height+1)[0] = 0;
	enerfyMin.at<Vec3b>(0,bound.height+1)[1] = 0;
	enerfyMin.at<Vec3b>(0,bound.height+1)[2] = 0;
	
	
	enerfyMin.convertTo(enerfyMin, CV_32FC3, 1/255.0);
	Mat diss(enerfyMin.size(),CV_32FC3, Scalar(-5,-5,-5) );

	
	Rect boxTxt, graBox;
	bool graFlag = false;
	for (int i=1;i<bound.height+2;i+=1)
	{
		diss.at<Vec3f>(0,i)[0] = enerfyMin.at<Vec3f>(0,i)[0]-enerfyMin.at<Vec3f>(0,i-1)[0];
		diss.at<Vec3f>(0,i)[1] = enerfyMin.at<Vec3f>(0,i)[1]-enerfyMin.at<Vec3f>(0,i-1)[1];
		diss.at<Vec3f>(0,i)[2] = enerfyMin.at<Vec3f>(0,i)[2]-enerfyMin.at<Vec3f>(0,i-1)[2];
	}
	for (int i=1;i<bound.height+2;i+=1)
	{
		if (diss.at<Vec3f>(0,i)[0] == 1){
			boxTxt.x = bound.x;	
			boxTxt.y = bound.y+i-1;
			boxTxt.width = bound.width;}
			
		if (diss.at<Vec3f>(0,i)[0] == -1){	
			boxTxt.height = i-(boxTxt.y+1 - bound.y);
			textboxes.push_back(boxTxt);}
			
		if (diss.at<Vec3f>(0,i)[1] == 1){
			//cout<<"yahoo"<<endl;
			graBox.x = bound.x;	
			graBox.y = bound.y+i-1;
			graBox.width = bound.width;}
			
		if (diss.at<Vec3f>(0,i)[1] == -1){	
			//cout<<"yahoo ends"<<endl;
			graBox.height = i-(graBox.y+1 - bound.y);
			graphicsBoxes.push_back(graBox);
			//cout<<boxTxt<<endl;
			}
			
	}	
	//cout<<diss<<endl;
	
}


void tmpSearch::grapHBlock2(Rect bound)
{

	Mat unaryVal(Size(1,bound.height+2), CV_8UC3,Scalar(0,0,0)); 
	
	Mat tDunary = unaryval(bound).clone();
	
// down sampling ///////// 1

	for (int i=1;i<bound.height+1;i+=1) // top 
	{
		
		Rect valBox; // initial block
		valBox.x = bound.x;
		valBox.y = bound.y + i;
		valBox.width = bound.width;
		valBox.height = 1;

		unaryVal.at<Vec3b>(0,i)[0] = textEnrg(valBox)/bound.width ;
		unaryVal.at<Vec3b>(0,i)[1] = (graEnrg(valBox)/bound.width+ bagEnrg(valBox)/bound.width)/2;
		unaryVal.at<Vec3b>(0,i)[2] = bagEnrg(valBox)/bound.width;

	}
	
// down sampling 2/////////////////
//	Mat imgtmp = unaryval(bound);
//	resize( imgtmp, unaryVal, Size( 1, imgtmp.rows ) );

	resize(tDunary, tDunary, Size(),(double)1/p_size, (double)1/p_size, INTER_NEAREST);		
	//namedWindow( "Display window0", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window0", tDunary ); 
	//waitKey(0); 	
	Mat enerfyMin = engyMin(3,tDunary,100);
	
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", enerfyMin ); 
	//waitKey(0); 	
	resize(enerfyMin, enerfyMin, Size(),(double)p_size, (double)p_size, INTER_NEAREST);
	
	enerfyMin.at<Vec3b>(0,0)[0] = 0;
	enerfyMin.at<Vec3b>(0,0)[1] = 0;
	enerfyMin.at<Vec3b>(0,0)[2] = 0;

	enerfyMin.at<Vec3b>(0,bound.height+1)[0] = 0;
	enerfyMin.at<Vec3b>(0,bound.height+1)[1] = 0;
	enerfyMin.at<Vec3b>(0,bound.height+1)[2] = 0;
	
	
	enerfyMin.convertTo(enerfyMin, CV_32FC3, 1/255.0);
	Mat diss(enerfyMin.size(),CV_32FC3, Scalar(-5,-5,-5) );

	
	Rect boxTxt, graBox;
	bool graFlag = false;
	for (int i=1;i<bound.height+2;i+=1)
	{
		diss.at<Vec3f>(0,i)[0] = enerfyMin.at<Vec3f>(0,i)[0]-enerfyMin.at<Vec3f>(0,i-1)[0];
		diss.at<Vec3f>(0,i)[1] = enerfyMin.at<Vec3f>(0,i)[1]-enerfyMin.at<Vec3f>(0,i-1)[1];
		diss.at<Vec3f>(0,i)[2] = enerfyMin.at<Vec3f>(0,i)[2]-enerfyMin.at<Vec3f>(0,i-1)[2];
	}
	for (int i=1;i<bound.height+2;i+=1)
	{
		if (diss.at<Vec3f>(0,i)[0] == 1){
			boxTxt.x = bound.x;	
			boxTxt.y = bound.y+i-1;
			boxTxt.width = bound.width;}
			
		if (diss.at<Vec3f>(0,i)[0] == -1){	
			boxTxt.height = i-(boxTxt.y+1 - bound.y);
			textboxes.push_back(boxTxt);}
			
		if (diss.at<Vec3f>(0,i)[1] == 1){
			//cout<<"yahoo"<<endl;
			graBox.x = bound.x;	
			graBox.y = bound.y+i-1;
			graBox.width = bound.width;}
			
		if (diss.at<Vec3f>(0,i)[1] == -1){	
			//cout<<"yahoo ends"<<endl;
			graBox.height = i-(graBox.y+1 - bound.y);
			graphicsBoxes.push_back(graBox);
			//cout<<boxTxt<<endl;
			}
			
	}	
	//cout<<diss<<endl;
	
}

vector<Rect> tmpSearch::getGrahpBlock()
{
 return graphicsBoxes;
}

vector<Rect> tmpSearch::getTxtBlock()
{
 return textboxes;
}
/*
Rect tmpSearch::grapHBlock(Rect bound)
{

	double minScoreT = 1e10, textUnar=0, graUnar=0, bacUnar=0, bacUnarbntr=0, graUnarbntr=0, textUnarbntr=0,unaryScoreTnG1 = 0, pairwiseScore1 = 0, templtCost1=0; // large value bntr
	Rect boundout;

	//cout<<"expected cetroid ="<<cetroidExp<<endl;
	int stepSize=p_size, stride=5;

	for (int i=bound.y;i<bound.height;i+=1) // top 
	for (int j=bound.height;j>bound.y;j-=1) // bottom 
	if (i<j )//and (l-k)>(j-i)
	{
		int wid=bound.width,high = j-i;

		float ppc = wid*high/(2*wid*stride+2*high*stride+4*stride*stride);
		
		Rect valBox; // initial block
		valBox.x = bound.x;
		valBox.y = i;
		valBox.width = wid;
		valBox.height = high;

		textUnar = textEnrg(valBox);
		graUnar = graEnrg(valBox);
		bacUnar = bagEnrg(valBox);
	
		valBox += Size(2*stride, 2*stride);
		valBox -= Point(stride,stride);
	
	
		bacUnarbntr = bagEnrg(valBox) - bacUnar;
		graUnarbntr = graEnrg(valBox) - graUnar;
		textUnarbntr= textEnrg(valBox) - textUnar;	
	

		unaryScoreTnG1 = .5*bacUnar +textUnar - graUnar;
			pairwiseScore1 = ppc*(textUnarbntr-graUnarbntr + .5*bacUnarbntr);	
		templtCost1 = (1-lambda)*unaryScoreTnG1 + lambda*pairwiseScore1;	

		if (templtCost1<minScoreT)
		{
			minScoreT=templtCost1;
			boundout = valBox;
		}

		
	}	

	return boundout;
}

*/

//00000000000000000000000
double tmpSearch2::unaryVal(Rect bound, int chn)
{
	double tl = Unaries[chn].at<double>(bound.y,bound.x);
	double tr = Unaries[chn].at<double>(bound.y,bound.x + bound.width);	
	double bl = Unaries[chn].at<double>(bound.y + bound.height,bound.x);
	double br = Unaries[chn].at<double>(bound.y + bound.height,bound.x + bound.width);
	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
		return  br-bl-tr+tl;
	else
		return 0;
}

Mat rectPrior1(Mat layout)
{
// 
	tmpSearch models(layout);
	// making template 1
	Rect bound = models.initialBlock();
	bound = models.refineBlock(bound);
	Rect bound2 = models.blockWithOut(bound);
	bound2 = models.refineBlock(bound2);
	//stage 2 segmentation one dimension
	models.grapHBlock(bound);
	models.grapHBlock(bound2);
	//stage 2 segmentation two dimension
	//models.grapHBlock2(bound);
	//models.grapHBlock2(bound2);
		
	vector<Rect> graxzx = models.getGrahpBlock();
	
	vector<Rect> txtbk = models.getTxtBlock();
	
	//cout<<"size = "<<graxzx.size()<<endl;
	//cout<<"txtbk size = "<<txtbk.size()<<endl;
	//Rect bound2 = models.blockWithOut(bound);
	//bound2 = models.refineBlock(bound2);
	// making template 2
	Rect bound3 = models.headingBlock(bound,bound2);
	// taking graphics box
	//Rect bound4 = models.grapHBlock(bound);
	//Rect bound5 = models.grapHBlock(bound2);
	//bound3 = models.refineBlock(bound3);
	Mat plotImg(layout.size(), CV_8UC3,Scalar(0,0,255)); 
	for (int i=0;i<graxzx.size();i++)
		rectangle(plotImg, graxzx[i], Scalar(0,255,0), -1, 8, 0 );
	for (int i=0;i<txtbk.size();i++)
		rectangle(plotImg, txtbk[i], Scalar(255,0,0), -1, 8, 0 );

	//for (int i=0;i<graxzx.size();i++)
	//	cout<<"gra = "<<graxzx[i]<<endl;
	//for (int i=0;i<txtbk.size();i++)
	//	cout<<"txtbk = "<<txtbk[i]<<endl;
	//rectangle(plotImg, bound2, Scalar(255,0,0), -1, 8, 0 );
	rectangle(plotImg, bound3, Scalar(0,255,255), -1, 8, 0 );
	//rectangle(plotImg, bound4, Scalar(0,255,0), -1, 8, 0 );
	//rectangle(plotImg, bound5, Scalar(0,255,0), -1, 8, 0 );
return 	plotImg;
}

//--------------------------------------------------------------------------------------------------

// Optimization XY-cut approach

vector<Rect> OptXYcutV(Mat layout)
{
	

	int imgHight=layout.rows,imgWidth = layout.cols;
	Mat laySplit[3], splitforI[3];
	split(layout, laySplit);
	split(layout, splitforI);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", layout ); 
	//waitKey(0); 
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if (layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[2])
				laySplit[0].at<uchar>(j,i)=1;
				else laySplit[0].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[0] and layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[2])
				laySplit[1].at<uchar>(j,i)=1;
				else laySplit[1].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[0])
				laySplit[2].at<uchar>(j,i)=1;	
				else laySplit[2].at<uchar>(j,i)=0;							
		}
	
	Mat plotImg(imgHight, imgWidth, CV_8UC1,Scalar(1)); // for ploting purpose
	// making foreground 
	Mat foreground;
	bitwise_or(laySplit[0],laySplit[1],foreground); 
	cv::Mat verticalSum;
	cv::reduce(foreground, verticalSum, 0, CV_REDUCE_SUM, CV_64FC1); //0-> vertical sum 1-> horizontal sum
	float lambda = 100;
	Mat cumVertSum;
	integral(verticalSum,cumVertSum,CV_64F);
	//double temp=1000;
	int pos=0;
	int widthh=1;
	//cout<<cumVertSum<<endl;
	//cout<<verticalSum<<endl;  verticalSum.rows - j + 
	//cout<<verticalSum.at<double>(0,200)<<endl;
	//cout<<cumVertSum.at<double>(1,200)<<endl;
	vector<Rect> bloklist;
	while (widthh>0){
	//for(int k=0;k<3;k++){
	float temp=1000000;
	for(int i=0;i<imgWidth;i++)
		for(int j=i;j<imgWidth;j++)
			if(cumVertSum.at<double>(1,j)>=0 and cumVertSum.at<double>(1,i)>=0 ){
			float cost =float(imgWidth-(j-i)) + float(cumVertSum.at<double>(1,j)-cumVertSum.at<double>(1,i))*lambda/imgHight;
			//cout<<"cost ="<<cost<<endl;
			if (cost<temp){
			temp=cost;
			pos=i;
			//cout<<"cost ="<<cost<<endl;
			widthh=j-i;}}
	//cout<<"pos ="<<pos;
	//cout<<", widthh ="<<widthh<<endl;
	Rect  bound;
	bound.x=pos;
	bound.y=0;
	bound.width=widthh;
	bound.height=imgHight;
	rectangle(plotImg, bound, Scalar(0), -1, 8, 0 );
	//bloklist.push_back(bound);
	//cout<<"image size = "<<laySplit[0].size()<<endl;
	//cout<<"bound size = "<<bound<<endl;
	
	//cout<<verticalSum.cols<<endl;	
	for (int i=0; i<widthh; i++)
		cumVertSum.at<double>(1,pos+i)=-1;
	//cout<<"cumVertSum.at<double>(1,pos+i)= "<<cumVertSum.at<double>(1,pos)<<endl;	
	}
	bloklist = detBlok(plotImg);
	//cv::groupRectangles(bloklist, 1, 0.2);
	//Mat vieww(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255));
	//for (int i=0;i<bloklist.size();i++)
	//	{
	//	rectangle(vieww, bloklist[i], Scalar(255,0,0), 5, 8, 0 );
	//	cout<<bloklist[i]<<endl;
		//}
	//cout<<"image size = "<<bloklist.size()<<endl;	
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", vieww ); 
	
	//waitKey(0); 		

			//Mat histSum = verticalSum(cv::Range(0, 2), cv::Range::all());
			//cout<<sum(histSum).val[0]<<endl;
	return bloklist;
}

//--------------------------------------------------------------------------------------
vector<Rect> OptXYcutH(Mat layout1)
{
	
	Mat layout = layout1.t();
	int imgHight=layout.rows,imgWidth = layout.cols;
	Mat laySplit[3], splitforI[3];
	split(layout, laySplit);
	split(layout, splitforI);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", layout ); 
	//waitKey(0); 
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if (layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[2])
				laySplit[0].at<uchar>(j,i)=1;
				else laySplit[0].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[0] and layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[2])
				laySplit[1].at<uchar>(j,i)=1;
				else laySplit[1].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[0])
				laySplit[2].at<uchar>(j,i)=1;	
				else laySplit[2].at<uchar>(j,i)=0;							
		}
	
	Mat plotImg(imgWidth, imgHight, CV_8UC1,Scalar(1)); // for ploting purpose
	// making foreground 
	Mat foreground;
	bitwise_or(laySplit[0],laySplit[1],foreground); 
	cv::Mat verticalSum;
	cv::reduce(foreground, verticalSum, 0, CV_REDUCE_SUM, CV_64FC1); //0-> vertical sum 1-> horizontal sum
	float lambda = 100;
	Mat cumVertSum;
	integral(verticalSum,cumVertSum,CV_64F);
	//double temp=1000;
	int pos=0;
	int widthh=1;
	//cout<<cumVertSum<<endl;
	//cout<<verticalSum<<endl;  verticalSum.rows - j + 
	//cout<<verticalSum.at<double>(0,200)<<endl;
	//cout<<cumVertSum.at<double>(1,200)<<endl;
	vector<Rect> bloklist;
	while (widthh>0){
	//for(int k=0;k<3;k++){
	float temp=1000000;
	for(int i=0;i<imgWidth;i++)
		for(int j=i;j<imgWidth;j++)
			if(cumVertSum.at<double>(1,j)>=0 and cumVertSum.at<double>(1,i)>=0 ){
			float cost =float(imgWidth-(j-i)) + float(cumVertSum.at<double>(1,j)-cumVertSum.at<double>(1,i))*lambda/imgHight;
			//cout<<"cost ="<<cost<<endl;
			if (cost<temp){
			temp=cost;
			pos=i;
			//cout<<"cost ="<<cost<<endl;
			widthh=j-i;}}
	//cout<<"pos ="<<pos;
	//cout<<", widthh ="<<widthh<<endl;
	Rect  bound;
	bound.x=0;
	bound.y=pos;
	bound.width=imgHight;
	bound.height= widthh;
	rectangle(plotImg, bound, Scalar(0), -1, 8, 0 );
	//bloklist.push_back(bound);
	//cout<<"image size = "<<laySplit[0].size()<<endl;
	//cout<<"bound size = "<<bound<<endl;
	
	//cout<<verticalSum.cols<<endl;	
	for (int i=0; i<widthh; i++)
		cumVertSum.at<double>(1,pos+i)=-1;
	//cout<<"cumVertSum.at<double>(1,pos+i)= "<<cumVertSum.at<double>(1,pos)<<endl;	
	}
	bloklist = detBlok(plotImg);
	//cv::groupRectangles(bloklist, 1, 0.2);
	//Mat vieww(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255));
	//for (int i=0;i<bloklist.size();i++)
		//{
		//rectangle(vieww, bloklist[i], Scalar(255,0,0), -1, 8, 0 );
		//cout<<bloklist[i]<<endl;
		//}
	//cout<<"image size = "<<bloklist.size()<<endl;	
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", vieww ); 
	
	//waitKey(0); 		

			//Mat histSum = verticalSum(cv::Range(0, 2), cv::Range::all());
			//cout<<sum(histSum).val[0]<<endl;
	return bloklist;
}
//======================================================================================
vector<Rect> detBlok(Mat binary)
{

	threshold(binary,binary, 0, 1, CV_THRESH_OTSU);
    // Fill the labelImage with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    	Mat labelImage;
	vector<Rect> bloklist;
	binary.convertTo(labelImage,  CV_32SC1);
	int labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage.rows; y++)
	        for(int x=0; x < labelImage.cols; x++) 
	        {
            		if(labelImage.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage, cv::Point(x,y), labelCount, &rect, 0, 0, 8);
		        bloklist.push_back(rect);

            		labelCount++;
        	}

    	return bloklist;
}

Mat OptXYcut(Mat prbImg)
{
	int imgHight=prbImg.rows,imgWidth = prbImg.cols;
	vector<Rect> vblock, hblock, fineH;
	hblock = OptXYcutH(prbImg);
	//vblock = OptXYcutV(prbImg);
	Mat vieww(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255));
	for (int i=0;i<hblock.size();i++){
		//rectangle(vieww, hblock[i], Scalar(255,255,0), -1, 8, 0 );
		Mat crped = prbImg(hblock[i]);
		vblock = OptXYcutV(crped);
		for (int j=0;j<vblock.size();j++){
			vblock[j].y = hblock[i].y;
			Mat crpd1 = prbImg(vblock[j]);
			fineH = OptXYcutH(crpd1);
			for (int k=0;k<fineH.size();k++){
				fineH[k].x += vblock[j].x;
				fineH[k].y += vblock[j].y;
				rectangle(vieww, fineH[k], Scalar(255,0,0), -1, 8, 0 );
				}
			
			}
		}			
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", vieww ); 
	
	//waitKey(0); 		
		
	return vieww;
}

////////////// Mat constrainedGC(Mat prbImg);// /////////////////////////////

Mat constrainedGC(Mat prbImg)
{
	//prbImg = prbImg.t();
	int imgHight=prbImg.rows,imgWidth = prbImg.cols;
	// making foreground and background
	Mat laySplit[3];
	split(prbImg, laySplit);
	Mat background = laySplit[2].clone();
	Mat foreground = laySplit[1].clone();
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
			if (prbImg.at<Vec3b>(j,i)[0]>prbImg.at<Vec3b>(j,i)[1]){
			foreground.at<uchar>(j,i)=laySplit[0].at<uchar>(j,i);	
			}
	//---------------------------------------------------
	// making integral image of foreground and background
	Mat isumbackground, isumforeground;
	background = (cv::Scalar::all(255)-background);
	foreground = (cv::Scalar::all(255)-foreground);
	integral(background,isumbackground,CV_64F);
	integral(foreground,isumforeground,CV_64F);
	//---------------------------------------------------
	//making pairwise cost
	//Mat pa
	cout<<foreground.size()<<endl;
	copyMakeBorder(foreground,foreground,1,0,0,0,BORDER_CONSTANT,Scalar(0));
	copyMakeBorder(foreground,foreground,0,1,0,0,BORDER_CONSTANT,Scalar(0));
	cout<<foreground.size()<<endl;
	//strarting horizontal cut
	double costt=100000000;
	int a=0,b,c,d;
	Rect boxes,box;
	for (int i=0;i<imgHight-3;i++){ //x
		double r1= isumbackground.at<double>(i,imgWidth);
	for (int j=i;j<imgHight-2;j++){//y
		double r2= isumforeground.at<double>(j,imgWidth)-isumforeground.at<double>(i,imgWidth);
	for (int k=j;k<imgHight-1;k++){
		double r3= isumbackground.at<double>(k,imgWidth)-isumbackground.at<double>(j,imgWidth);
	for (int l=k;l<imgHight;l++){
		double r4= isumforeground.at<double>(l,imgWidth)-isumforeground.at<double>(k,imgWidth);	
		double r5= isumbackground.at<double>(imgHight,imgWidth)-isumbackground.at<double>(l,imgWidth);	
		double energy = r1+r2+r3+r4+r5;
		//cout<<" "<<r1<<", "<<r2<<", "<<r3<<", "<<r4<<", "<<r5<<endl;
		if (energy<costt){
		costt=energy;
		//cout<<"energy = "<<energy<<endl;
		boxes.x=0;
		boxes.y=i;
		boxes.width=imgWidth;
		boxes.height=j-i;
		box.x=0;
		box.y=k;
		box.width=imgWidth;
		box.height=l-k;		
		}
	//cout<<"exit"<<endl;	
	//a=a+1;	
	}}}}	
	cout<<a<<endl;
	rectangle(prbImg, boxes, Scalar(0,0,0), -1, 8, 0 );
	rectangle(prbImg, box, Scalar(0,0,0), -1, 8, 0 );			
	namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	imshow( "Display window", prbImg ); 
	waitKey(0); 		
		
	return prbImg;
}
//====================================================================================
Mat rectPrior20(Mat layout, Mat layout1, string temp_folder)
{
    	string filename;
	    vector<layoutElement> output; double temp=2; int kkk;
  	    tmpSearch2 models(layout, layout1);
	    vector<layoutElement> layoutPrior;  
        vector<string> files = vector<string>();
	    getdir(temp_folder,files);
	    for (unsigned int i = 0;i < files.size();i++) // files.size()
	    {   
            string inputPath= temp_folder + string("/") + files[i] ;
            cout<<"yahooo ="<<inputPath<<endl;
	        Mat image = imread(inputPath, CV_LOAD_IMAGE_COLOR);
	        resize(image, image, layout.size(), INTER_NEAREST);
            // crating rectangular list with primary labels (t,g,b)    
            layoutPrior = models.CreatlayoutElementVecFmImg(image);  
            layoutPrior = models.energyCal(layoutPrior);    
            cout<<files[i]<<" - "; 
	        if (temp>layoutPrior[0].cost)
	    	    {temp = layoutPrior[0].cost;
	    	    output = layoutPrior;
	    	    kkk = i;
                //  cout<<"temp = "<<temp<<endl;
	    	    }
            cout<<"energy = "<<layoutPrior[0].cost<<endl;    
	        layoutPrior.clear();            
                   
        }   
	    filename = files[kkk];
	    cout<<"op = "<<filename<<endl;         
        Mat OutImg = models.CreatImgFmlayoutElementVec(output);
    return OutImg;
}

tmpSearch2::tmpSearch2(Mat layout, Mat layout1) 
{
	//unaryval = layout.clone();
	split(layout, laySplit);
	//invert
	laySplit[0]=Scalar::all(255)-laySplit[0];
	laySplit[1]=Scalar::all(255)-laySplit[1];
	laySplit[2]=Scalar::all(255)-laySplit[2];
	
	integral(laySplit[0],Unaries[0],CV_64F); // text
	integral(laySplit[2],Unaries[1],CV_64F); // background
	integral(laySplit[1],Unaries[2],CV_64F); // block diagrams 	
    //laySplit.release();
    // split(layout1, laySplit);
	// //invert
	// laySplit[0]=Scalar::all(255)-laySplit[0];
	// laySplit[1]=Scalar::all(255)-laySplit[1];
	// laySplit[2]=Scalar::all(255)-laySplit[2];
	
	// integral(laySplit[0],Unaries[3],CV_64F); // Drowing
	// integral(laySplit[1],Unaries[4],CV_64F); // graphs
	// integral(laySplit[2],Unaries[5],CV_64F); // natural images
       
    
	imgHight=layout.rows,imgWidth = layout.cols;
	lambda = 0.0; gama=10; // gama (1---100)
	
}


int getdir (string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) 
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) 
	{
		if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 )
		{
    		//cout<<dirp->d_name<<"hohohhhh"<<endl;
			files.push_back(string(dirp->d_name));	
		}

	}
	closedir(dp);
	return 0;
}

vector<layoutElement> tmpSearch2::CreatlayoutElementVecFmImg(Mat image)
{
    vector<layoutElement> layoutPrior;
    layoutElement blk1;
	Mat laySplit[3];
	split(image, laySplit);
	for (int j=0;j<3;j++)
		threshold(laySplit[j],laySplit[j],125,1,THRESH_BINARY);	
//-----------------------------------------------------------------------------------	
	cv::Mat labelImage1, labelImage2;
	laySplit[0].convertTo(labelImage1,  CV_32SC1);
	int labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage1.rows; y++)
	        for(int x=0; x < labelImage1.cols; x++) 
	        {
            		if(labelImage1.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage1, cv::Point(x,y), labelCount, &rect, 0, 0, 8);	
		        blk1.box = rect;
		        blk1.label = 1; //text region
		        layoutPrior.push_back(blk1);
		        labelCount++;
		    }       
	laySplit[1].convertTo(labelImage2,  CV_32SC1);
	labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage2.rows; y++)
	        for(int x=0; x < labelImage2.cols; x++) 
	        {
            		if(labelImage2.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage2, cv::Point(x,y), labelCount, &rect, 0, 0, 8);	
		        blk1.box = rect;
		        blk1.label = 2;// graphics region
		        layoutPrior.push_back(blk1);
		        labelCount++;
		}   	
	
    return layoutPrior;
}

vector<layoutElement> tmpSearch2::energyCal(vector<layoutElement> layoutPrior)
 {
 	Mat plotImg(laySplit[0].size(), CV_8UC3,Scalar(0,0,255)); 
 	int textEng=0, graphicsE=0, backE=0;//, notxtBlk=0, nograBlock=0;
 	double texBlkPairCost = 0.0, graBlkPairCost = 0.0 ;
	//rectangle(plotImg, bound, Scalar(255,0,0), -1, 8, 0 );
	//cout<<"size = "<<layoutPrior.size()<<endl;
	for (int i=0;i<layoutPrior.size();i++)
	{
		
		backE = backE+unaryVal(layoutPrior[i].box, 1); // 1-> background
		if (layoutPrior[i].label==1)
		{
			rectangle(plotImg, layoutPrior[i].box, Scalar(255,0,0), -1, 8, 0 );
			textEng = textEng+unaryVal(layoutPrior[i].box,0); //0->text
			//notxtBlk++;
			//texBlkPairCost = textPairEnergy(layoutPrior[i].box);
		}
		if (layoutPrior[i].label==2)
		{
			rectangle(plotImg, layoutPrior[i].box, Scalar(0,255,0), -1, 8, 0 );
            double temp=1.0;
            for(int j=2;j<3;j++)
                if (temp<unaryVal(layoutPrior[i].box,j))	
                    {
                        temp = unaryVal(layoutPrior[i].box,j);
                        layoutPrior[i].label=j;
                    }
           graphicsE = graphicsE+temp;	                             
		}		
	}	
	double unaryCost = textEng + graphicsE + Unaries[1].at<double>(imgHight,imgWidth) - backE;
	unaryCost = unaryCost/(imgHight*imgWidth*255);
	//cout<<"unary cost  = "<<unaryCost<<endl;
	//double pairCost= (texBlkPairCost+0.0001) + (graBlkPairCost+0.0001);
	//cout<<"pair cost  = "<<pairCost<<endl;
	double totalCost = unaryCost;// + lambda*pairCost;
    layoutPrior[0].cost = unaryCost;
	//cout<<"total cost  = "<<totalCost<<endl;
	//cout<<"imgHight= "<<imgHight<<endl;
	//cout<<"imgWidth = "<<imgWidth<<endl;
	//double pp = textPairEnergy(layoutPrior[1].box);
	//double pp1 = grapPairEnergy(layoutPrior[1].box);
	
	//cout<<"gra = "<<graphicsE<<endl;
	
	// namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	// imshow( "Display window", plotImg ); 
	// waitKey(0);
	return layoutPrior; 
    
 }

Mat tmpSearch2::CreatImgFmlayoutElementVec(vector<layoutElement> layoutPrior)
{
    Mat outImg(imgHight, imgWidth, CV_8UC3,Scalar(127,2,240));//background, 
	for (int i=0;i<layoutPrior.size();i++)
        if (layoutPrior[i].label==1) // 1-> text  
            rectangle(outImg,layoutPrior[i].box, Scalar(127,201,127),-1);                                                                        
        else if (layoutPrior[i].label==2) // 2-> block diagrams  
            rectangle(outImg,layoutPrior[i].box, Scalar(190,174,212),-1);
        else if (layoutPrior[i].label==3) // 3-> drowing  
            rectangle(outImg,layoutPrior[i].box, Scalar(253,192,134),-1);
        else if (layoutPrior[i].label==4) // 4-> graphics  
            rectangle(outImg,layoutPrior[i].box, Scalar(255,255,153),-1);
        else if (layoutPrior[i].label==5) // 5-> natural images  
            rectangle(outImg,layoutPrior[i].box, Scalar(56,108,176),-1);                                    
    return outImg;
}

























































































Mat rectPrior4(Mat layout) // two rectangle of text and graphics
{
// without assuming width and height
	int imgHight=layout.rows,imgWidth = layout.cols;
// splitting the out put
	Mat laySplit[3];
	split(layout, laySplit);
	namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	imshow( "Display window", layout ); 
	waitKey(0); 	
//making 1 and zero
//	for (int i=0;i<3;i++)
//		threshold(laySplit[i],laySplit[i],125,1,THRESH_BINARY);	
//	Mat plotImg(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255)); // for ploting purpose
//	int cnt=0,delta=5;float costParam = .5;
// making of integral image for fast computation
//unary term
	Mat isumText, isumGraphics, isumback;
	integral(laySplit[0],isumText,CV_64F);
	integral(laySplit[1],isumGraphics,CV_64F);
	integral(laySplit[2],isumback,CV_64F);
// pair wise term
	int ddepth = CV_16S;
	Mat grad_x[3], grad_y[3], totalGrad;	
	for (int i=0;i<3;i++) {
		Sobel( laySplit[i], grad_x[i], ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
		Sobel( laySplit[i], grad_y[i], ddepth, 0, 1, 3, 1, 0, BORDER_DEFAULT );
		}

	totalGrad = abs(grad_x[0])+abs(grad_x[1])+abs(grad_x[2]) + abs(grad_y[0])+abs(grad_y[1])+abs(grad_y[2]);
	convertScaleAbs( totalGrad, totalGrad );// to the viewing purpose!
	Mat pairWise;
	integral(totalGrad,pairWise,CV_64F);
//-----------------	
	double minScoreG = 1e10,minScoreT = 1e10,textUnar=0,graUnar=0,bacUnar=0, unaryScoreGra=0,pairwiseScore, topPair, botPair, leftPair, ritePair,score,scoreText,unaryScoreText,scoreGra; // large value
	float lambda = 0.6;
	int stepSize=100, cnt=0;
	Rect bound, grapbound;
	for (int i=0;i<imgHight-1;i+=stepSize) // top 
	for (int j=imgHight;j>1;j-=stepSize) // bottom
	for (int k=0;k<imgWidth-1;k+=stepSize) // left
	for (int l=imgWidth;l>1;l-=stepSize) // right
	if (i<j and k<l)
	{
		double tl= isumText.at<double>(i,k);
		double tr= isumText.at<double>(i,l);	
		double bl= isumText.at<double>(j,k);
		double br= isumText.at<double>(j,l);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			textUnar = br-bl-tr+tl;
		tl= isumGraphics.at<double>(i,k);
		tr= isumGraphics.at<double>(i,l);	
		bl= isumGraphics.at<double>(j,k);
		br= isumGraphics.at<double>(j,l);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			graUnar = br-bl-tr+tl;
		tl= isumback.at<double>(i,k);
		tr= isumback.at<double>(i,l);	
		bl= isumback.at<double>(j,k);
		br= isumback.at<double>(j,l);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			bacUnar = br-bl-tr+tl;	
		unaryScoreGra = textUnar + bacUnar - graUnar;	
		unaryScoreText = graUnar + bacUnar - textUnar;
		// find pairwise cost
		tl= pairWise.at<double>(i,k);
		tr= pairWise.at<double>(i,l);	
		bl= pairWise.at<double>(i+1,k);
		br= pairWise.at<double>(i+1,l);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			topPair = br-bl-tr+tl;	
		tl= pairWise.at<double>(j,k);
		tr= pairWise.at<double>(j,l);	
		bl= pairWise.at<double>(j+1,k);
		br= pairWise.at<double>(j+1,l);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			botPair = br-bl-tr+tl;	
		tl= pairWise.at<double>(i,k);
		tr= pairWise.at<double>(i,k+1);	
		bl= pairWise.at<double>(j,k);
		br= pairWise.at<double>(j,k+1);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			leftPair = br-bl-tr+tl;	
		tl= pairWise.at<double>(i,l);
		tr= pairWise.at<double>(i,l+1);	
		bl= pairWise.at<double>(j,l);
		br= pairWise.at<double>(j,l+1);
		if (br>=0 and bl>=0 and tr>=0 and tl>=0)
			ritePair = br-bl-tr+tl;	
		pairwiseScore = (topPair+botPair+leftPair+ritePair);	
		scoreGra = (1-lambda)*unaryScoreGra + lambda*pairwiseScore; 								
		scoreText = (1-lambda)*unaryScoreText + lambda*pairwiseScore; 
		if (scoreGra<minScoreG)
		{
			minScoreG=scoreGra;
			grapbound.x = k;
			grapbound.y = i;
			grapbound.width = l-k;
			grapbound.height = j-i;
		}
//-------------------------------------------------------------------------------------------------------
		// for text
		if (scoreText<minScoreT)
		{
			minScoreT=scoreText;
			bound.x = k;
			bound.y = i;
			bound.width = l-k;
			bound.height = j-i;
		}
		
		
	}	
	//cout<<"unaryScoreGra = "<<unaryScoreGra<<endl;
	//cout<<"pairwiseScore = "<<pairwiseScore<<endl;
	//cout<<"unaryScoreText = "<<unaryScoreText<<endl;
	//cout<<"pairwiseScore = "<<pairwiseScore<<endl;

	double min, max;
	minMaxLoc(totalGrad, &min, &max);
	
	//cout<<"max = "<<bound<<endl;
	Mat plotImg(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255)); // for ploting purpose
	rectangle(plotImg, bound, Scalar(255,0,0), -1, 8, 0 );
	rectangle(plotImg, grapbound, Scalar(0,255,0), -1, 8, 0 );
	namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	imshow( "Display window", plotImg ); 
	waitKey(0);	
	
return 	plotImg;
}


Mat rectPrior2(Mat layout)
{
// without assuming width and height
	int imgHight=layout.rows,imgWidth = layout.cols;
// splitting the out put
	Mat laySplit[3];
	split(layout, laySplit);
	namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	imshow( "Display window", layout ); 
	waitKey(0); 	
//making 1 and zero
	for (int i=0;i<3;i++)
		threshold(laySplit[i],laySplit[i],125,1,THRESH_BINARY);	
	Mat plotImg(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255)); // for ploting purpose
	int cnt=0,delta=5;float costParam = .5;
// making of integral image for fast computation
	Mat isumText, isumGraphics;
	integral(laySplit[0],isumText,CV_64F);
	//integral(laySplit[1],isumGraphics,CV_64F);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", laySplit[0] ); 
	//waitKey(0); 	
//---------------------------------------------------------------------------------------------------------------------------------	
	bool iter=true;
	// for testing purpose
	//int j=400,i=350;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[0].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				// initial density check
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta and widthCost<costParam*tempRect.height*delta and heightCost<costParam*tempRect.width*delta and jcost<costParam*tempRect.width*delta)
				      		iter=false;
			      	}

			      	//cout<<"the rect width is = "<<tempRect.width<<endl;
			      	rectangle(laySplit[0],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(255,0,0),-1);
			//cnt++;
			//cout<<"yahoooooooooooo "<<cnt<<endl;	
			//cout<<"yaho entered for text "<<endl;	
			}
			iter=true;
			
		}
//--------------------------------------------------------------------------------	
	
	integral(laySplit[1],isumText,CV_64F);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", isumText ); 
	//waitKey(0); 	
	//integral(laySplit[1],isumGraphics,CV_64F);
//---------------------------------------------------------------------------------------------------------------------------------	
	//bool 
	iter=true;
	// for testing purpose
	//int j=400,i=350;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[1].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				// initial density check
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta and widthCost<costParam*tempRect.height*delta and heightCost<costParam*tempRect.width*delta and jcost<costParam*tempRect.width*delta)
				      		iter=false;
			      	}

			      	//cout<<"the rect width is = "<<tempRect.width<<endl;
			      	rectangle(laySplit[1],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(0,255,0),-1);
			//cout<<"yaho entered for image "<<endl;	
			//cnt++;
			}
			iter=true;
			
		}
//--------------------------------------------------------------------------------		
		
	return plotImg;
}
//--------------------------------------------------------------------------------
// The function rectPrior3 is used/replase the graph cut segmentation 
// it segment the image with obtimal region growing rectangular prior
// input is the image with unary potensial as pixel value ranges (0-255)
//---------------------------------------------------
Mat rectPrior3(Mat layout)
{
	int imgHight=layout.rows,imgWidth = layout.cols;
	Mat laySplit[3], splitforI[3];
	split(layout, laySplit);
	split(layout, splitforI);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", layout ); 
	//waitKey(0); 
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if (layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[0]>layout.at<Vec3b>(j,i)[2])
				laySplit[0].at<uchar>(j,i)=1;
				else laySplit[0].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[0] and layout.at<Vec3b>(j,i)[1]>layout.at<Vec3b>(j,i)[2])
				laySplit[1].at<uchar>(j,i)=1;
				else laySplit[1].at<uchar>(j,i)=0;
			if (layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[1] and layout.at<Vec3b>(j,i)[2]>layout.at<Vec3b>(j,i)[0])
				laySplit[2].at<uchar>(j,i)=1;	
				else laySplit[2].at<uchar>(j,i)=0;							
		}
	
	Mat plotImg(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255)); // for ploting purpose
	int cnt=0,delta=5;float costParam = .5;
// making of integral image for fast computation
	Mat isumText, isumGraphics;
	integral(splitforI[0],isumText,CV_64F);
//---------------------------------------------------------------------------------------------------------------------------------	
	bool iter=true;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[0].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height*.5)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta*255)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta*255)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta*255)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta*255)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta*255 and widthCost<costParam*tempRect.height*delta*255 and heightCost<costParam*tempRect.width*delta*255 and jcost<costParam*tempRect.width*delta*255)
				      		iter=false;
			      	}
			      	rectangle(laySplit[0],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(255,0,0),-1);
			}
			iter=true;
			
		}
//--------------------------------------------------------------------------------	
	integral(splitforI[1],isumText,CV_64F);
	iter=true;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[1].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				// initial density check
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta*255)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta*255)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta*255)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta*255 and widthCost<costParam*tempRect.height*delta*255 and heightCost<costParam*tempRect.width*delta*255 and jcost<costParam*tempRect.width*delta*255)
				      		iter=false;
			      	}
			      	rectangle(laySplit[1],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(0,255,0),-1);
			}
			iter=true;
		}
//--------------------------------------------------------------------------------		
	return plotImg;
}
//
//---------------------------------------------------------------------------------
Mat GridGraph_Individually(int num_labels,Mat img,int lambda)
{

	int height=img.rows;//HEIGHT
	int width=img.cols;//width
	int num_pixels=height*width;


	int *result = new int[num_pixels];   // stores result of optimization
	int rw;
	int col;
	Mat  opimage =img.clone();
//image is transformed int 1 drow in row major order

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually


		for ( int i = 0; i < num_pixels; i++ )
		{
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;

			}	
			else
			{
			rw=(i+1)/width;
			col=((i+1)%width)-1;
			}

			int blue=img.at<uchar>(rw,col);
			//int green=img.at<cv::Vec3b>(rw,col)[1];
			//int red=img.at<cv::Vec3b>(rw,col)[2];



			for (int l = 0; l < num_labels; l++ )
			{
				if(l==0)
					 gc->setDataCost(i,l,(255-blue)/*+red+green*/);
			 	if(l==1)
			 		gc->setDataCost(i,l,(blue)/*+red+blue*/);
		 		//if(l==2)
		 			//gc->setDataCost(i,l,(255-red)/*+blue+green*/);

			}
		}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{

				if(l1==l2)
				//int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
				gc->setSmoothCost(l1,l2,0);

				else

				gc->setSmoothCost(l1,l2,lambda);


			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());


		

		for ( int  i = 0; i < num_pixels; i++ )
		{
			result[i] = gc->whatLabel(i);
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;
			}
			else
			{
				rw=(i+1)/width;
				col=((i+1)%width)-1;
			}
			if(result[i]==0) //sky
			{
		//cout<<"label 0 \n";
				opimage.at<uchar>(rw,col)=255;//blue
				//opimage.at<cv::Vec3b>(rw,col)[1]=0;
				//opimage.at<cv::Vec3b>(rw,col)[2]=0;
			}
			if(result[i]==1) // grass
			{
			opimage.at<uchar>(rw,col)=0;
			//opimage.at<cv::Vec3b>(rw,col)[1]=255;
			//opimage.at<cv::Vec3b>(rw,col)[2]=0;
			//cout<<"label 1 \n";
			}
			//if(result[i]==2) //third object
			//{
			//	opimage.at<cv::Vec3b>(rw,col)[0]=0;
			//	opimage.at<cv::Vec3b>(rw,col)[1]=0;
			//	opimage.at<cv::Vec3b>(rw,col)[2]=255;//red
			//}
		}





		//imwrite( "outputimage.png", opimage );


		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}
	delete [] result;
	return opimage;
}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void splitStr(const string& s, char c, vector<string>& v) {
   string::size_type i = 0;
   string::size_type j = s.find(c);

   while (j != string::npos) {
      v.push_back(s.substr(i, j-i));
      i = ++j;
      j = s.find(c, j);

      if (j == string::npos)
         v.push_back(s.substr(i, s.length()));
   }
}
