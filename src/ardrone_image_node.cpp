#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// For quaternions
#include <Eigen/Geometry> 

// Include FPL libraries
#include <ceigen.h>

#include "ardrone_image_node.h"

#include <eigen3/Eigen/Core>

#include <CameraSensor.h>
#include <CameraModel.h>

#if HAS_CVARS
#  include <CVars/CVar.h>
#endif

#include <fiducials/tracker.h>

#define MAX_FILENAME 1024

namespace enc = sensor_msgs::image_encodings;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
void constImageProcessingCallback( const sensor_msgs::ImageConstPtr& image_msg )
{
    ROS_INFO( "Received image, size: %dx%d", image_msg->width, image_msg->height );
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvShare( image_msg, enc::BGR8 );
    }
    catch( cv_bridge::Exception& e ) {
        ROS_ERROR( "cv_bridge exception: %s", e.what() );
        return;
    }

    cv::imshow( "ardrone_image", cv_ptr->image );
    cv::waitKey( 3 );
}

////////////////////////////////////////////////////////////////////////////////
// Track a grid (will lead to ambiguous pose estimation)
void ARDroneImageNode::_imageProcessingCallbackGrid( const sensor_msgs::ImagePtr& image_msg )
{
    char sImageName[MAX_FILENAME];
    static int nNumCapturedImages = 0;

    //ROS_INFO( "Received image, size: %dx%d", image_msg->width, image_msg->height );
    cv_bridge::CvImagePtr cv_ptr;

    try {
        //cv_ptr = cv_bridge::toCvCopy( image_msg, enc::BGR8 );
        cv_ptr = cv_bridge::toCvCopy( image_msg, enc::MONO8 );
    }
    catch( cv_bridge::Exception& e ) {
        ROS_ERROR( "cv_bridge exception: %s", e.what() );
        return;
    }

    cv::Rect r( 0, 0, 160, 120 );
    cv::Mat img_gray_cropped( 120, 160, CV_8UC1, cv::Scalar(0) );
    cv_ptr->image(r).convertTo( img_gray_cropped, CV_8UC1, 1, 0 );

    int nKey = cv::waitKey( 3 ); 

    if( (char)nKey == 's' ) {
        snprintf( sImageName, MAX_FILENAME, "view_%03d.png", nNumCapturedImages );
        cv::imwrite( sImageName, img_gray_cropped );
        nNumCapturedImages++;
        ROS_INFO( "Saving image, %s, size: %dx%d", sImageName,
                  img_gray_cropped.cols, img_gray_cropped.rows );
    }

    Eigen::MatrixXd mGridPoints;
    CvSize board_size = {5,3};
    IplImage aImage = img_gray_cropped;
    IplImage* pImage = &aImage;
    mGridPoints = CEIGEN::FindChessboardCorners( pImage, board_size, true );
    bool bFound = mGridPoints.size() > 0; 

    if( bFound ) {
        // Comes after saving to avoid having grid drawn on the image
        Eigen::MatrixXf mGridPointsF = mGridPoints.cast<float>();
        cvDrawChessboardCorners( pImage, board_size, 
                                 reinterpret_cast<CvPoint2D32f*>( mGridPointsF.data() ),
                                 mGridPointsF.cols(), bFound );
    }

    cv::imshow( "ardrone_image", img_gray_cropped );

    if( !bFound ) {
        return;
    }

    // 3D Grid
    Eigen::MatrixXd m_mGrid = CEIGEN::make_grid( board_size.height, board_size.width );

    // Compute position
    Eigen::Matrix3d mR;
    Eigen::Vector3d mt;
    bool bCompute = compute_extrinsics_non_lin( m_CameraModel, 
                                                m_mGrid, mGridPoints,
                                                mR, mt );

    // Invert to obtain pose instead of transform
    Eigen::Matrix3d mR_inv = mR.transpose();
    Eigen::Vector3d mt_inv = -mR.transpose() * mt;

    if( !bCompute ) {
        ROS_ERROR( "ERROR: computing extrinsics\n" );
        return;
    }

    cout << "Z: " << endl << mt(2) << endl;
    cout << "Det: " << endl << mR.determinant() << endl;

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.seq = image_msg->header.seq;
    //pose_msg.header.frame_id = "/ardrone_low_cam";
    pose_msg.header.frame_id = "/map";
    pose_msg.header.stamp = image_msg->header.stamp;
    pose_msg.pose.position.x = mt_inv(0);
    pose_msg.pose.position.y = mt_inv(1);
    pose_msg.pose.position.z = mt_inv(2);
    Eigen::Quaternion<double> q( mR_inv );
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    m_PubPose.publish( pose_msg );
}

////////////////////////////////////////////////////////////////////////////////
// Track ellipses
void ARDroneImageNode::_imageProcessingCallbackEllipses( const sensor_msgs::ImagePtr& image_msg )
{
    char sImageName[MAX_FILENAME];
    static int nNumCapturedImages = 0;

    //ROS_INFO( "Received image, size: %dx%d", image_msg->width, image_msg->height );
    cv_bridge::CvImagePtr cv_ptr;

    try {
        //cv_ptr = cv_bridge::toCvCopy( image_msg, enc::BGR8 );
        cv_ptr = cv_bridge::toCvCopy( image_msg, enc::MONO8 );
    }
    catch( cv_bridge::Exception& e ) {
        ROS_ERROR( "cv_bridge exception: %s", e.what() );
        return;
    }

    cv::Rect r( 0, 0, 160, 120 );
    cv::Mat img_gray_cropped( 120, 160, CV_8UC1, cv::Scalar(0) );
    cv_ptr->image(r).convertTo( img_gray_cropped, CV_8UC1, 1, 0 );

    int nKey = cv::waitKey( 3 ); 

    if( (char)nKey == 's' ) {
        snprintf( sImageName, MAX_FILENAME, "view_%03d.png", nNumCapturedImages );
        cv::imwrite( sImageName, img_gray_cropped );
        nNumCapturedImages++;
        ROS_INFO( "Saving image, %s, size: %dx%d", sImageName,
                  img_gray_cropped.cols, img_gray_cropped.rows );
    }

    ////////////////////////////////////////////////////////////////////////////
    //const int nImageWidth = image_msg->width;
    //const int nImageHeight = image_msg->height;
    const int nImageWidth  = 160;
    const int nImageHeight = 120;

    if( m_pTracker.get() == NULL ) {
        // Create ellipse tracker
        unique_ptr<Tracker> aT( new Tracker( nImageWidth, nImageHeight ) );
        m_pTracker = move( aT );
        m_pTracker.get()->target.GenerateRandom
            ( 60,25/(842.0/297.0),75/(842.0/297.0),15/(842.0/297.0),Eigen::Vector2d(297,210) );
        m_pTracker.get()->target.SaveEPS( "target.eps" );
    }

    if( m_pCam.get() == NULL ) {
        unique_ptr<CCameraModel::CameraModel> pCameraModelUndist
            ( m_CameraModel.new_rectified_camera( nImageWidth, nImageHeight ) );

        // Camera parameters for undistorted image
        unique_ptr<LinearCamera> aC
            ( new LinearCamera( nImageWidth, nImageHeight,
                                pCameraModelUndist->get<double>( "fx" ),
                                pCameraModelUndist->get<double>( "fy" ),
                                pCameraModelUndist->get<double>( "cx" ),
                                pCameraModelUndist->get<double>( "cy" ) ) );
        m_pCam = move( aC );
 
        cout << pCameraModelUndist->get<double>( "fx" ) << " " <<
            pCameraModelUndist->get<double>( "fy" ) << " " <<
            pCameraModelUndist->get<double>( "cx" ) << " " <<
            pCameraModelUndist->get<double>( "cy" ) << endl;
    }

    cv::Mat imageUndist( nImageHeight, nImageWidth, CV_8UC1 );

    IplImage aImage = img_gray_cropped;

    fig.imshow( &aImage );
    fig.draw();

    
    IplImage image_undist = imageUndist;

    m_CameraModel.undistort_image( nImageWidth, nImageHeight, nImageWidth,
                                   reinterpret_cast<const unsigned char*>( aImage.imageData ),
                                   reinterpret_cast<unsigned char*>( image_undist.imageData ), 
                                   !m_bLUTComputed );
    m_bLUTComputed = true;

    fig_un.imshow( imageUndist );

    const bool bTrackingGood =
        m_pTracker->ProcessFrame( *m_pCam.get(), (unsigned char*)image_undist.imageData );
    if ( bTrackingGood ) {
        for( size_t i=0; i<m_pTracker->conics.size(); ++i ) {
            const Eigen::Vector2d vCenter = m_pTracker->conics[i].center;
            std::vector<std::pair<double, double> > vCenter2(2);
            vCenter2.push_back( std::pair<double, double>( vCenter[0], vCenter[1] ) );
            
            fig_un.plot( vCenter2, "r+" );
        }
    }

    fig_un.draw();
    nKey = fig.wait( 10 );

    // Send back computer transform
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.seq = image_msg->header.seq;
    //pose_msg.header.frame_id = "/ardrone_low_cam";
    pose_msg.header.frame_id = "/map";
    pose_msg.header.stamp = image_msg->header.stamp;

    Eigen::Matrix<double,4,4> mT = m_pTracker->T_gw.matrix();

    pose_msg.pose.position.x = mT(0,3)/100;
    pose_msg.pose.position.y = mT(1,3)/100;
    pose_msg.pose.position.z = mT(2,3)/100;
    Eigen::Matrix<double,3,3> mR = mT.block(0,0,3,3);
    Eigen::Quaternion<double> q( mR );
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    m_PubPose.publish( pose_msg );

#if 0
    Eigen::MatrixXd mGridPoints;
    CvSize board_size = {5,3};
    IplImage aImage = img_gray_cropped;
    IplImage* pImage = &aImage;
    mGridPoints = CEIGEN::FindChessboardCorners( pImage, board_size, true );
    bool bFound = mGridPoints.size() > 0; 

    if( bFound ) {
        // Comes after saving to avoid having grid drawn on the image
        Eigen::MatrixXf mGridPointsF = mGridPoints.cast<float>();
        cvDrawChessboardCorners( pImage, board_size, 
                                 reinterpret_cast<CvPoint2D32f*>( mGridPointsF.data() ),
                                 mGridPointsF.cols(), bFound );
    }

    cv::imshow( "ardrone_image", img_gray_cropped );

    if( !bFound ) {
        return;
    }

    // 3D Grid
    Eigen::MatrixXd m_mGrid = CEIGEN::make_grid( board_size.height, board_size.width );

    // Compute position
    Eigen::Matrix3d mR;
    Eigen::Vector3d mt;
    bool bCompute = compute_extrinsics_non_lin( m_CameraModel, 
                                                m_mGrid, mGridPoints,
                                                mR, mt );

    // Invert to obtain pose instead of transform
    Eigen::Matrix3d mR_inv = mR.transpose();
    Eigen::Vector3d mt_inv = -mR.transpose() * mt;

    if( !bCompute ) {
        ROS_ERROR( "ERROR: computing extrinsics\n" );
        return;
    }

    cout << "Z: " << endl << mt(2) << endl;
    cout << "Det: " << endl << mR.determinant() << endl;

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.seq = image_msg->header.seq;
    //pose_msg.header.frame_id = "/ardrone_low_cam";
    pose_msg.header.frame_id = "/map";
    pose_msg.header.stamp = image_msg->header.stamp;
    pose_msg.pose.position.x = mt_inv(0);
    pose_msg.pose.position.y = mt_inv(1);
    pose_msg.pose.position.z = mt_inv(2);
    Eigen::Quaternion<double> q( mR_inv );
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    m_PubPose.publish( pose_msg );
#endif
}

////////////////////////////////////////////////////////////////////////////////
ARDroneImageNode::ARDroneImageNode() : 
    m_bInitialised( false ), m_bLUTComputed( false ),
    fig( "Image" ), fig_un( "Undist" )
{  
}

////////////////////////////////////////////////////////////////////////////////
ARDroneImageNode::~ARDroneImageNode() {}

////////////////////////////////////////////////////////////////////////////////
bool ARDroneImageNode::Init() {
    if( !SelectCamera() ) {
        return false;
    }
    m_SubImage = m_Node.subscribe( "/ardrone/image_raw", 1000, 
                                   &ARDroneImageNode::_imageProcessingCallbackEllipses, this );
    m_PubPose  = m_Node.advertise<geometry_msgs::PoseStamped>( "/ardrone/cam_pose", 1000 );
    
    string sCameraModelFile = "calibrated_ardrone_bottom.txt";
    if( !m_CameraModel.load( sCameraModelFile ) ) {
        ROS_ERROR( "ERROR: problem loading camera model from file: %s\n", sCameraModelFile.c_str() );
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool ARDroneImageNode::SelectCamera() {
    ros::ServiceClient client = m_Node.serviceClient<std_srvs::Empty>( "/mav/ToggleCam" );
    std_srvs::Empty srv;

    // FIXME: hack, we currently call toggle twice to obtain the camera pointing down,
    // TODO: write a service that takes in the camera type directly
    if( client.call( srv ) ) {
        ROS_INFO( "Sucessful toggle call " );
    }
    else {
        ROS_ERROR( "Failed to call service " );
        return false;
    }
    if( client.call( srv ) ) {
        ROS_INFO( "Sucessful toggle call " );
    }
    else {
        ROS_ERROR( "Failed to call service " );
        return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
void ARDroneImageNode::run() {
    if( !m_bInitialised ) {
        Init();
        m_bInitialised = true;
    }
    ros::spin();
}

////////////////////////////////////////////////////////////////////////////////
int main( int argc, char **argv )
{
    ros::init( argc, argv, "ardrone_image_node" );
    ARDroneImageNode().run();

    return 0;
}

