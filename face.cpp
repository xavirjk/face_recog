#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat initialImageProcessing(Mat frame);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade, eyes_cascade2;
string face_cascadeName = "../haarcascades/haarcascade_frontalface_alt2.xml";
string eyes_cascadeName1 = "../haarcascades/haarcascade_eye.xml";
string eyes_cascadeName2 = "../haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main(int argc, char *argv[])
{
    try
    {
        face_cascade.load(face_cascadeName);
        eyes_cascade.load(eyes_cascadeName1);
        eyes_cascade2.load(eyes_cascadeName2);
    }
    catch (Exception &e)
    {
        std::cerr << e.what() << '\n';
        if (face_cascade.empty())
        {
            cerr << " Error!! Could not load faceDetector." << endl;
            cerr << face_cascadeName << endl;
            exit(-1);
        }
    }

    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << " Error!! could not open Webcam" << endl;
        exit(-1);
    }

    for (;;)
    {
        Mat frame;

        capture >> frame;

        if (frame.empty())
        {
            cout << "No frame" << endl;
            break;
        }
        Mat img;
        const int DetectionWidth = 320;
        float scale = frame.cols / (float)DetectionWidth;

        if (frame.cols > DetectionWidth)
        {
            int scaledHeight = cvRound(frame.rows / scale);
            resize(frame, img, Size(DetectionWidth, scaledHeight));
        }
        else
            img = frame;
        Mat equilizedimg = initialImageProcessing(img);
        vector<Rect> faces;
        face_cascade.detectMultiScale(equilizedimg, faces, 1.1f, 4, CASCADE_SCALE_IMAGE, Size(20, 20));

        if (frame.cols > DetectionWidth)
        {
            for (int i = 0; i < faces.size(); i++)
            {
                int scaled_x, scaled_y, scaled_h, scaled_w;

                Point Pt1(faces[i].x, faces[i].y);
                Point Pt2(faces[i].height + faces[i].x, faces[i].width + faces[i].y);

                scaled_x = cvRound(faces[i].x * scale);
                scaled_y = cvRound(faces[i].y * scale);
                scaled_w = cvRound(faces[i].width * scale);
                scaled_h = cvRound(faces[i].height * scale);

                Point Pt3(scaled_x, scaled_y);
                Point Pt4(scaled_h + scaled_x, scaled_w + scaled_y);

                rectangle(img, Pt1, Pt2, Scalar(0, 255, 0), 2, 8, 0);
                rectangle(frame, Pt3, Pt4, Scalar(0, 255, 0), 2, 8, 0);

                Mat faceROI = equilizedimg(faces[i]);
                vector<Rect> eyes;
                //-- In each face, detect eyes
                eyes_cascade.detectMultiScale(faceROI, eyes, 1.1f, 4, CASCADE_SCALE_IMAGE, Size(20, 20));
                for (int j = 0; j < eyes.size(); j++)
                {
                    Point center1(faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5);
                    Point center2(scaled_x + cvRound(eyes[j].x * scale) + cvRound(eyes[j].width * scale) * 0.5, scaled_y + cvRound(eyes[j].y * scale) + cvRound(eyes[j].height * scale) * 0.5);
                    int radius1 = cvRound((eyes[j].width + eyes[j].height) * 0.2);
                    int radius2 = cvRound((cvRound(eyes[j].width * scale) + cvRound(eyes[j].height * scale)) * 0.2);
                    circle(img, center1, radius1, Scalar(255, 0, 0), 4, 8, 0);
                    circle(frame, center2, radius2, Scalar(255, 0, 0), 4, 8, 0);
                }
            }
        }
        imshow("Equilizedimage", equilizedimg);
        imshow("original", img);
        imshow("ORfarme", frame);

        char c = (char)waitKey(25);

        if (c == 27)
            break;
    }
    return 0;
}

Mat initialImageProcessing(Mat frame)
{
    Mat gray;
    if (frame.channels() == 3)
        cvtColor(frame, gray, COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cvtColor(frame, gray, COLOR_BGRA2GRAY);
    else
        gray = frame;
    Mat equilizedImage;
    equalizeHist(gray, equilizedImage);
    return equilizedImage;
}