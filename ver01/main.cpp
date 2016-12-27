#include <opencv2/opencv.hpp>
#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"
#include "HandDetector.hpp"
#include "GCoptimization.h"
#include "LinkedBlockList.h"	
#include <string>
#include "rectprior.h"


using namespace std;
using namespace cv;


//@@@@@@@@@@@@@@@@@@@@@ read all file from the folder@@@@@@@@@@@@@@@@@@@@@@
// int getdir (string dir, vector<string> &files)
// {
// 	DIR *dp;
// 	struct dirent *dirp;
// 	if((dp  = opendir(dir.c_str())) == NULL) 
// 	{
// 		cout << "Error(" << errno << ") opening " << dir << endl;
// 		return errno;
// 	}

// 	while ((dirp = readdir(dp)) != NULL) 
// 	{
// 		if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 )
// 		{
//     		//cout<<dirp->d_name<<"hohohhhh"<<endl;
// 			files.push_back(string(dirp->d_name));	
// 		}

// 	}
// 	closedir(dp);
// 	return 0;
// }


int main (int argc, char * const argv[]) 
{
    
    
    // Mat jobs = 
   	int Rnadval = 6; 
	bool TRAIN_MODEL = 0;
	bool TEST_MODEL  = 1;
	bool CHECK = 0;
	
	int target_width = 1000;						// for resizing the input (small is faster)
	
	// maximum number of image masks that you will use
	// must have the masks prepared in advance
	// only used at training time
	int num_models_to_train = 30;	
	
	
	// number of models used to compute a single pixel response
	// must be less than the number of training models
	// only used at test time
	int num_models_to_average = 30;
	
	// runs detector on every 'step_size' pixels
	// only used at test time
	// bigger means faster but you lose resolution
	// you need post-processing to get contours
	int step_size = 1;				
	
	// Assumes a certain file structure e.g., /root/img/basename/00000000.jpg
	//string root = "/_GTA/";
	//string basename = "Yin_American";
	//string img_prefix		= root + "img/"		+ basename + "/";			// color images /mnt/data/datasets/ieeePaper/gtImg
	//string msk_prefix		= root + "mask/"	+ basename + "/";			// binary masks
	//string model_prefix		= root + "models/"	+ basename + "/";			// output path for learned models
	//string globfeat_prefix  = root + "globfeat/"+ basename + "/";			// output path for color histograms
	
	string basename = "Documents";
	string img_prefix		= "/users/jobinkv/2nd_data/word100img/originalImage/";// color images 
	string msk_prefix		= "/users/jobinkv/2nd_data/word100img/groundTruth/";// binary masks
	string validOpImg		= "/users/jobinkv/2nd_data/word100img/validationOp/";
    	string testImgLoc		= "/users/jobinkv/2nd_data/word100img/originalImage/";
    	string opImgLoc			= "/users/jobinkv/2nd_data/word100img/outputs/mrf_30/";
    	string temp_folder      	= "/mnt/data/datasets/ieeePaper/cvprTemplates/template";
	//string opImgLoc		= "/mnt/data/datasets/ieeePaper/100ImageTrainNtestSet/rectPriorOutPut/";

	stringstream setName;
	setName.str("");
	setName << "Exp_"<<Rnadval<<"/"; //WtGlobalFet"; // onlyRF
	string pg_code = setName.str();
	
	string textModel		= "./models/loc/"+pg_code;			// output path for learned models
	string textModelG  		= "./models/glb/"+pg_code;			// output path for color histograms
	
		
	// types of features to use (you will over-fit if you do not have enough data)
	// r: RGB (5x5 patch)
	// v: HSV
	// l: LAB
	// b: BRIEF descriptor
	// o: ORB descriptor
	// s: SIFT descriptor
	// u: SURF descriptor
	// h: HOG descriptor
	string feature_set = "rb";
	
	if(CHECK)
	{
		vector<string> outFiles;
		
		stringstream ss;
		ifstream readFile;
		getdir(opImgLoc,outFiles);
		for (unsigned int i = 0;i < outFiles.size();i++){
			ss.str("");
			ss <<opImgLoc<<outFiles[i];
			//cout <<ss.str()<< endl;
			readFile.open (ss.str().c_str(), ifstream::in);
			string line;
			getline(readFile,line);
			outFiles[i].erase (outFiles[i].end()-10, outFiles[i].end());// remove .word.txt to get the image name
			ss.str("");
			ss <<img_prefix<<outFiles[i];
			//cout <<ss.str()<< endl;
			Mat image = imread(ss.str(),1);	
			while(readFile){
				string word;
				list<int> wordLst;
				istringstream iss(line);
				while(iss){
					iss>>word;
					int val;
					val=atoi(word.c_str());
			//		cout<<val<<endl;
					wordLst.push_back(val);}
				Rect box;
				box.x=wordLst.front() ;
				wordLst.pop_front();
				box.y=wordLst.front() ;
				wordLst.pop_front();
				box.width=wordLst.front() ;
				wordLst.pop_front();
				box.height=wordLst.front() ;
				rectangle(image,box,Scalar(0),4,8);
				cout<<box<<endl;
				getline(readFile,line);}
			readFile.close();	
			namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
    			imshow( "Display window", image );                   // Show our image inside it.
    		 	waitKey(0);	
    		 				
			//outFiles[i].erase (outFiles[i].end()-10, outFiles[i].end());// remove .word.txt to get the image name
			//cout <<outFiles[i]<< endl;	
			}			
	
	}	
	vector<string> totalFilenames;	
	vector<string> trainFiles;	
	vector<string> validationFiles;	
	vector<string> testFiles;	
	getdir(msk_prefix,totalFilenames);
	srand(Rnadval);// chage the number will result defferent trainFiles, validationFiles and testFiles. 
	random_shuffle ( totalFilenames.begin(), totalFilenames.end() );
	int setNo =0;
	trainFiles.reserve(30); 
	validationFiles.reserve(30); 
	testFiles.reserve(40); 
	copy(totalFilenames.begin()+setNo,totalFilenames.begin()+setNo+30,back_inserter(trainFiles));
	copy(totalFilenames.begin()+setNo+30,totalFilenames.begin()+setNo+60,back_inserter(validationFiles));
	copy(totalFilenames.begin()+setNo+60,totalFilenames.begin()+setNo+100,back_inserter(testFiles));
	
	if(TRAIN_MODEL)
	{
		HandDetector text;
		text.loadMaskFilenames(trainFiles);
		text.trainModels(basename, img_prefix, msk_prefix,textModel,textModelG,feature_set,num_models_to_train,target_width,0);
	
	}
	
	
	stringstream ss;
	if(TEST_MODEL)	{
		HandDetector text;//, grap, back;
		text.testInitialize(textModel,textModelG,feature_set,num_models_to_average,target_width);
		stringstream optloc;
		optloc.str("");
		optloc << "mkdir -p "<<validOpImg<<"stage_"<<Rnadval;
		system(optloc.str().c_str());
		for (unsigned int i = 0;i < validationFiles.size();i++) {
			ss.str("");
			ss <<testImgLoc<<validationFiles[i];
			cout <<ss.str()<< endl;
			Mat image = imread(ss.str(),1);		 
			Mat im = image.clone();
			cout<<"testing"<<endl;	
			text.test(im,num_models_to_average,step_size);
			Mat combinePrb, segOut ;
			Mat raw_prob[3];
			text.colormap(text._response_img,combinePrb,1);		
			stringstream outname;
			outname.str("");
			outname <<validOpImg<<"stage_"<<Rnadval<<"/"<<validationFiles[i]<<".words.txt";
			cout<<optloc.str()<<endl;
			int num_labels = 2;
			int lambada=2*255;
			segOut = GridGraph_Individually(num_labels,combinePrb,lambada);
			std::ofstream wordCord;    //create output file
 			wordCord.open(outname.str().c_str(),std::ios::app);
			threshold(segOut,segOut, 0, 1, CV_THRESH_OTSU);	
			cv::Mat labelImage;
			segOut.convertTo(labelImage,  CV_32SC1);
			int labelCount = 2; // starts at 2 because 0,1 are used already
			for(int y=0; y < labelImage.rows; y++)
	        	for(int x=0; x < labelImage.cols; x++) {
            			if(labelImage.at<int>(y,x)!= 1)
                			continue;
		        	cv::Rect rect;	
		        	cv::floodFill(labelImage, cv::Point(x,y), labelCount, &rect, 0, 0, 8);	
		        	wordCord<<rect.x<<"\t"<<rect.y<<"\t"<<rect.width<<"\t"<<rect.height<<"\n";}		
			combinePrb.release();	
            		im.release();		
			int SHOW_RAW_PROBABILITY = 0;
			if(SHOW_RAW_PROBABILITY){
				Mat raw_prob[3];
				text.colormap(text._response_img,raw_prob[0],0);		
				imshow("text prb",raw_prob[0]);	}}
		}

}
