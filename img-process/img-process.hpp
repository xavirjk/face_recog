#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
Mat processImage(Point leftEye, Point rightEye, Mat faceROI);
void equilizefaceHalves(Mat &faceImg);
