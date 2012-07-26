#ifndef ARDRONE_IMAGE_NODE_H
#define ARDRONE_IMAGE_NODE_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>

// Include FPL libraries
#include <CameraModel.h>
#include <COpenCV.h>

class Tracker;
class LinearCamera;

class ARDroneImageNode {
 public:
    ARDroneImageNode();
    ~ARDroneImageNode();
    
    bool Init();
    bool SelectCamera();

    void run();

 private:
    bool m_bInitialised;
    bool m_bLUTComputed;
    CCameraModel::CameraModel m_CameraModel;

    ros::NodeHandle m_Node;
    ros::Subscriber m_SubImage;
    ros::Publisher  m_PubPose;

    geometry_msgs::PoseStamped m_CamPose;
    std::unique_ptr<Tracker> m_pTracker; // ellipse tracker
    std::unique_ptr<LinearCamera> m_pCam;  

    COPENCV::Figure fig;
    COPENCV::Figure fig_un;

 private:
    void _imageProcessingCallbackGrid( const sensor_msgs::ImagePtr& image_msg );
    void _imageProcessingCallbackEllipses( const sensor_msgs::ImagePtr& image_msg );

};

#endif
