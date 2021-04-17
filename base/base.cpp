#include <base.hpp>
using namespace faceRecog::init;

faceRecognizer::faceRecognizer()
{
    try
    {
        face_cascade.load(face_cascadeName);
        eyes_cascade1.load(eyes_cascadeName1);
        eyes_cascade2.load(eyes_cascadeName2);
    }
    catch (Exception &e)
    {
        cerr << e.what() << '\n';
        if (face_cascade.empty())
        {
            cerr << " Error!! Could not load faceDetector." << endl;
            cerr << face_cascadeName << endl;
            exit(-1);
        }
    }
    initWebcam();
    r_mode = FACES_COLLECT_MODE;
}
faceRecognizer::~faceRecognizer() {}

void faceRecognizer::initWebcam()
{
    try
    {
        capture.open(0);
    }
    catch (Exception &e)
    {
        cerr << e.what() << "endl";
    }
    if (!capture.isOpened())
    {
        cerr << " Error!! could not open Webcam" << endl;
        exit(-1);
    }
}
void faceRecognizer::recognizer()
{
    for (;;)
    {
        Mat frame;
        capture >> frame;

        if (frame.empty())
        {
            cout << "No frame" << endl;
            break;
        }
        Mat img = scaleImage(frame);
        Mat gray = gray_ScaledImage(img);
        Mat equilizedImage;

        equalizeHist(gray, equilizedImage);

        face_cascade.detectMultiScale(equilizedImage, faces, 1.1f, 4, CASCADE_FIND_BIGGEST_OBJECT, Size(20, 20));

        for (int i = 0; i < faces.size(); i++)
        {
            drawFaces(frame, img, i);
            Mat faceROI = equilizedImage(faces[i]);
            faceRect = Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
            vector<Rect> eyes;
            //-- In each face, detect eyes
            detectTheEyes(faceROI, leftEye, rightEye, eyes_cascade1, eyes_cascade2);
            if (leftEye.x > 0 && rightEye.x > 0)
            {
                drawEyes(img, i);
                preProcessedFace = processImage(leftEye, rightEye, faceROI);
                imshow("Image", img);
                imshow("masked", preProcessedFace);

                switch (r_mode)
                {
                case STARTUP_MODE:
                    cout << "Startup" << endl;
                    break;
                case FACES_COLLECT_MODE:
                    collectFaces(frame);
                    break;
                case TRAINING_MODE:
                    trainModel();
                    break;
                case RECOG_MODE:
                    recognizeface();
                    break;
                default:
                    break;
                }
            }
        }
        char c = (char)waitKey(25);

        if (c == 27)
            break;
        imshow("frame", frame);
    }
}

void faceRecognizer::collectFaces(Mat &frame)
{
    if (preProcessedFace.data)
    {
        double IMAGE_DIFF = 10000000000.0;
        if (previous_preprocessed_face.data)
        {
            IMAGE_DIFF = getSimilarity(preProcessedFace, previous_preprocessed_face);
        }
        double currentTime = (double)getTickCount();
        double TIME_DIFF = (currentTime - pre_time) / getTickFrequency();

        if ((IMAGE_DIFF > 0.3) && (TIME_DIFF > 1.0))
        {
            Mat mirroredFace;
            flip(preProcessedFace, mirroredFace, 1);

            preprocessedFaces.push_back(preProcessedFace);
            preprocessedFaces.push_back(mirroredFace);
            labels.push_back(_identity);
            labels.push_back(_identity);

            previous_preprocessed_face = preProcessedFace;
            pre_time = currentTime;
            rectangle(frame, faceRect, Scalar(0, 255, 0), 2, 8, 0);

            if (preprocessedFaces.size() % 50 == 0)
            {

                String input;
                cout << "Enter y to REPEAT the step for a different set, x to proceed training the model:" << endl;
                cin >> input;
                if (input == "y")
                {
                    _identity += 1;
                    r_mode = FACES_COLLECT_MODE;
                }
                else if (input == "x")
                {
                    r_mode = TRAINING_MODE;
                }
                //r_mode = TRAINING_MODE;
            }

            cout << "no:" << preprocessedFaces.size() << endl;
        }
    }
}

void faceRecognizer::trainModel()
{
    cout << "Faces collected...Initializing Training mode " << endl;
    model = EigenFaceRecognizer::create();
    bool sufficientData = true;
    if (preprocessedFaces.size() <= 0 || preprocessedFaces.size() != labels.size())
    {
        cout << "Warning! Insufficient Data for Training...Collect more Data" << endl;
        sufficientData = false;
    }
    if (sufficientData)
    {
        if (model.empty())
        {
            cerr << "Error!! The face recognizer Algorithm is not Present in your openCv version" << endl;
            exit(-1);
        }
        model->train(preprocessedFaces, labels);
        showTrainingDebugData(model, 70, 70);
        r_mode = RECOG_MODE;
    }
    else
    {
        r_mode = FACES_COLLECT_MODE;
    }
}

void faceRecognizer::recognizeface()
{
    //Face Recog from the Model
    Mat reconstructedFace;
    reconstructedFace = reconstructFace(model, preProcessedFace);
    if (reconstructedFace.data)
    {
        imshow("reconstructed face", reconstructedFace);
        double similarity = getSimilarity(preProcessedFace, reconstructedFace);
        string outputString;

        if (similarity < UNKNOWN_PERSON_THRESHOLD)
        {
            int id = model->predict(preProcessedFace);
            outputString = to_string(id);
        }
        else
        {
            outputString = "UNKNOWN";
        }
        cout << "Identity: " << outputString << " Similarity: " << similarity << endl;
        r_mode = STARTUP_MODE;
        model->save("trainedModel.yml");

        //get confidence Ratio
    }
    else
    {
        cout << "Retrying recognition.....Stay still" << endl;
    }
}
Mat faceRecognizer::scaleImage(Mat frame)
{
    scale = frame.cols / (float)DetectionWidth;
    if (frame.cols > DetectionWidth)
    {
        int scaledHeight = cvRound(frame.rows / scale);
        resize(frame, frame, Size(DetectionWidth, scaledHeight));
    }
    return frame;
}

Mat faceRecognizer::gray_ScaledImage(Mat frame)
{
    Mat gray;

    if (frame.channels() == 3)
        cvtColor(frame, gray, COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cvtColor(frame, gray, COLOR_BGRA2GRAY);
    else
        gray = frame;
    return gray;
}
void faceRecognizer::drawFaces(Mat &frame, Mat &img, int i)
{
    int scaled_x, scaled_y, scaled_h, scaled_w;
    scaled_x = cvRound(faces[i].x * scale);
    scaled_y = cvRound(faces[i].y * scale);
    scaled_w = cvRound(faces[i].width * scale);
    scaled_h = cvRound(faces[i].height * scale);
    Point Pt1(faces[i].x, faces[i].y);
    Point Pt2(faces[i].height + faces[i].x, faces[i].width + faces[i].y);

    Point Pt3(scaled_x, scaled_y);
    Point Pt4(scaled_h + scaled_x, scaled_w + scaled_y);

    rectangle(img, Pt1, Pt2, Scalar(0, 255, 0), 2, 8, 0);
    rectangle(frame, Pt3, Pt4, Scalar(0, 255, 0), 2, 8, 0);
}

void faceRecognizer::drawEyes(Mat &img, int i)
{
    Point center1(faces[i].x + leftEye.x, faces[i].y + leftEye.y);
    Point center2(faces[i].x + rightEye.x, faces[i].y + rightEye.y);
    circle(img, center1, 15, Scalar(255, 0, 0), 4, 8, 0);
    circle(img, center2, 15, Scalar(255, 0, 0), 4, 8, 0);
}

void faceRecognizer::setMode(int mode)
{
    switch (mode)
    {
    case 0:
        r_mode = STARTUP_MODE;
        break;
    case 1:
        r_mode = FACES_COLLECT_MODE;
        break;
    case 2:
        r_mode = TRAINING_MODE;
        break;
    case 3:
        r_mode = RECOG_MODE;
        break;
    default:
        r_mode = STARTUP_MODE;
        break;
    }
}