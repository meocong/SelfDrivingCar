//test lane detection
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "api_lane_detection.h"
#include <iostream>
#include <fstream>

#define video_frame_width 640
#define video_frame_heigh 480
using namespace cv;
using namespace std;

int main(int argc, char ** args) {
	VideoCapture cap;
	cap.open(args[1]);

	bool is_write_data = true; 
	ofstream myfile;
	myfile.open ("center_point_data.txt");			//open file to write data
	if (is_write_data) 								//initial
	{
		myfile <<"Data of center point : "<<std::endl;  	
	  	myfile <<"x  y"<<std::endl;  	
	}
	

	bool is_save_file = true;
	string gray_filename = "gray_center.avi";
	VideoWriter gray_videoWriter;
	int codec = CV_FOURCC('D','I','V', 'X');
	Size output_size(video_frame_width, video_frame_heigh);
	if(is_save_file)
    {
        gray_videoWriter.open(gray_filename, codec, 24, output_size, false);
    }


	bool is_show_cam = true;
	Mat frame;		
	Mat binImage;
	MSAC msac;
	Rect roi1 = Rect(0, video_frame_heigh*3/4, video_frame_width, video_frame_heigh/4);
    api_vanishing_point_init( msac );


	while (true) {
		Point center_point(0,0);
		Mat center_frame;
		cap >> frame;
		if (frame.channels() == 3) cvtColor(frame, center_frame, CV_BGR2GRAY);
		else center_frame = frame.clone();

		api_get_vanishing_point( frame, roi1, msac, center_point, is_show_cam,"Wavelet");
		int erosion_size = 1;
   		Mat element = cv::getStructuringElement( MORPH_RECT, Size( 2*erosion_size + 1, 2*erosion_size+1 ),
													Point( erosion_size, erosion_size ) );
		edgeProcessing(frame(roi1), binImage, element, "Wavelet");

		//show video
		if(is_show_cam)
        {
			//show binary image
			if (!binImage.empty()) imshow("bin",binImage);

			//show center_frame image
			circle(center_frame, center_point, 5, Scalar(0,255,0), 2,8,0);
			imshow("center_frame", center_frame);	

			//show frame image
			imshow("frame",frame);
		}

		if(is_write_data)
		{
			myfile << center_point.x<<" "<<center_point.y<<std::endl;
		}
		
		//write center gray video
		if(is_save_file)
        {
            if(!center_frame.empty())
                gray_videoWriter.write(center_frame);
		}
		waitKey(10);
	}

	//release videowriter
	if(is_save_file)
    {
        gray_videoWriter.release();
	}	

	//release file
	myfile.close();

	return 0;
}

