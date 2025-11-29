#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "depthai/depthai.hpp"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <mutex>
#include <thread>

constexpr float FPS = 60.0f;
const cv::Size SIZE(640, 480);

static std::vector<float> g_points;
static std::vector<unsigned char> g_colors;
static std::mutex g_mutex;

void buildPointArray(const cv::Mat& depth, const cv::Mat& rgb, float fx, float fy, float cx, float cy,
                     std::vector<float>& outPts, std::vector<unsigned char>& outCols)
{
    outPts.clear();
    outCols.clear();
    if (depth.empty())
        return;

    cv::Mat depth32f;
    depth.convertTo(depth32f, CV_32F);

    for (int y = 0; y < depth32f.rows; ++y)
    {
        const float* dptr = depth32f.ptr<float>(y);
        const cv::Vec3b* rgbPtr = rgb.empty() ? nullptr : rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < depth32f.cols; ++x)
        {
            float d = dptr[x];
            if (d <= 0.0f || d > 5000.0f)
                continue;
            d /= 1000.0f;
            float px = (x - cx) * d / fx;
            float py = (y - cy) * d / fy;
            float pz = d;
            outPts.push_back(px);
            outPts.push_back(py);
            outPts.push_back(pz);
            if (rgbPtr)
            {
                const cv::Vec3b& c = rgbPtr[x];
                outCols.push_back(c[2]);
                outCols.push_back(c[1]);
                outCols.push_back(c[0]);
            }
            else
            {
                outCols.push_back(255);
                outCols.push_back(255);
                outCols.push_back(255);
            }
        }
    }
}

struct Vec3
{
    float x, y, z;
};
static Vec3 g_camCenter = {0.0f, 0.0f, 0.0f};
static int g_lastX = 0, g_lastY = 0;
static bool g_drag = false;
static int g_lastButton = -1;
static int g_winW = 800, g_winH = 600;
static float g_camDist = 1.0f;
static float g_camYaw = 0.0f;
static float g_camPitch = 0.0f;

void displayGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)g_winW / (float)g_winH;
    gluPerspective(60.0, aspect, 0.01, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float yawRad = g_camYaw * (3.14159265f / 180.0f);
    float pitchRad = g_camPitch * (3.14159265f / 180.0f);
    float cp = cosf(pitchRad);
    Vec3 camPos;
    camPos.x = g_camCenter.x + g_camDist * cp * sinf(yawRad);
    camPos.y = g_camCenter.y + g_camDist * sinf(pitchRad);
    camPos.z = g_camCenter.z + g_camDist * cp * cosf(yawRad);
    Vec3 up;
    up.x = sinf(pitchRad) * sinf(yawRad);
    up.y = cosf(pitchRad);
    up.z = sinf(pitchRad) * cosf(yawRad);
    gluLookAt(camPos.x, camPos.y, camPos.z,
              g_camCenter.x, g_camCenter.y, g_camCenter.z,
              up.x, up.y, up.z);

    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0.2f, 0, 0);
    glColor3f(0, 1, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0.2f, 0);
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 0.2f);
    glEnd();

    std::lock_guard<std::mutex> lk(g_mutex);
    size_t nPts = g_points.size() / 3;
    glBegin(GL_POINTS);
    for (size_t i = 0; i < nPts; ++i)
    {
        unsigned char r = g_colors[i * 3 + 0];
        unsigned char g = g_colors[i * 3 + 1];
        unsigned char b = g_colors[i * 3 + 2];
        glColor3ub(r, g, b);
        glVertex3f(g_points[i * 3 + 0], g_points[i * 3 + 1], g_points[i * 3 + 2]);
    }
    glEnd();

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    g_winW = w;
    g_winH = h;
    glViewport(0, 0, w, h);
}

void mouseFunc(int button, int state, int x, int y)
{
    if (button == 3 && state == GLUT_DOWN)
    {
        g_camDist *= 0.9f;
        return;
    }
    if (button == 4 && state == GLUT_DOWN)
    {
        g_camDist *= 1.1f;
        return;
    }

    if (button == GLUT_LEFT_BUTTON)
    {
        g_drag = (state == GLUT_DOWN);
        if (state == GLUT_DOWN)
            g_lastButton = GLUT_LEFT_BUTTON;
        else
            g_lastButton = -1;
        g_lastX = x;
        g_lastY = y;
        if (state == GLUT_UP)
            g_drag = false;
        return;
    }

    if (button == GLUT_RIGHT_BUTTON)
    {
        g_drag = (state == GLUT_DOWN);
        if (state == GLUT_DOWN)
            g_lastButton = GLUT_RIGHT_BUTTON;
        else
            g_lastButton = -1;
        g_lastX = x;
        g_lastY = y;
        if (state == GLUT_UP)
            g_drag = false;
        return;
    }
}

void motionFunc(int x, int y)
{
    if (!g_drag)
        return;
    int dx = x - g_lastX;
    int dy = y - g_lastY;
    if (g_lastButton == GLUT_LEFT_BUTTON)
    {
        g_camYaw += dx * 0.5f;
        g_camPitch += dy * 0.5f;
        if (g_camPitch > 89.0f)
            g_camPitch = 89.0f;
        if (g_camPitch < -89.0f)
            g_camPitch = -89.0f;
    }
    else if (g_lastButton == GLUT_RIGHT_BUTTON)
    {
        float yawRad = g_camYaw * (3.14159265f / 180.0f);
        float pitchRad = g_camPitch * (3.14159265f / 180.0f);
        float cp = cosf(pitchRad);
        Vec3 camPos;
        camPos.x = g_camCenter.x + g_camDist * cp * sinf(yawRad);
        camPos.y = g_camCenter.y + g_camDist * sinf(pitchRad);
        camPos.z = g_camCenter.z + g_camDist * cp * cosf(yawRad);

        Vec3 forward;
        forward.x = g_camCenter.x - camPos.x;
        forward.y = g_camCenter.y - camPos.y;
        forward.z = g_camCenter.z - camPos.z;
        float flen = sqrtf(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
        if (flen > 1e-6f)
        {
            forward.x /= flen;
            forward.y /= flen;
            forward.z /= flen;
        }

        Vec3 worldUp = {0.0f, 1.0f, 0.0f};
        Vec3 right;
        right.x = forward.y * worldUp.z - forward.z * worldUp.y;
        right.y = forward.z * worldUp.x - forward.x * worldUp.z;
        right.z = forward.x * worldUp.y - forward.y * worldUp.x;
        float rlen = sqrtf(right.x * right.x + right.y * right.y + right.z * right.z);
        if (rlen > 1e-6f)
        {
            right.x /= rlen;
            right.y /= rlen;
            right.z /= rlen;
        }

        Vec3 up;
        up.x = right.y * forward.z - right.z * forward.y;
        up.y = right.z * forward.x - right.x * forward.z;
        up.z = right.x * forward.y - right.y * forward.x;

        float scale = 0.0015f * g_camDist;
        Vec3 delta;
        delta.x = (-dx) * scale * right.x + (dy)*scale * up.x;
        delta.y = (-dx) * scale * right.y + (dy)*scale * up.y;
        delta.z = (-dx) * scale * right.z + (dy)*scale * up.z;

        g_camCenter.x += delta.x;
        g_camCenter.y += delta.y;
        g_camCenter.z += delta.z;
    }
    g_lastX = x;
    g_lastY = y;
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 27 || key == 'q')
    {
        exit(0);
    }
}

static void timerFunc(int /*value*/)
{
    glutPostRedisplay();
    glutTimerFunc(1000 / FPS, timerFunc, 0);
}

int main()
{
    dai::Pipeline pipeline;

    auto node_camera_rgb = pipeline.create<dai::node::Camera>();
    auto node_camera_tof = pipeline.create<dai::node::ToF>();
    auto node_synchronize = pipeline.create<dai::node::Sync>();
    auto node_alignment = pipeline.create<dai::node::ImageAlign>();

    node_alignment->setRunOnHost(true);

    node_camera_tof->build(dai::CameraBoardSocket::CAM_A);
    node_camera_rgb->build(dai::CameraBoardSocket::CAM_C);

    node_synchronize->setSyncThreshold(std::chrono::milliseconds(static_cast<uint32_t>(1000 / FPS)));
    node_synchronize->setRunOnHost(true);

    auto node_camera_rgb_output = node_camera_rgb->requestOutput(std::make_pair(SIZE.width, SIZE.height), dai::ImgFrame::Type::RGB888i, dai::ImgResizeMode::CROP, FPS, true);

    node_camera_rgb_output->link(node_synchronize->inputs["rgb"]);
    node_camera_tof->depth.link(node_alignment->input);
    node_alignment->outputAligned.link(node_synchronize->inputs["depth_aligned"]);
    node_synchronize->inputs["rgb"].setBlocking(false);
    node_camera_rgb_output->link(node_alignment->inputAlignTo);
    auto syncQueue = node_synchronize->out.createOutputQueue();

    pipeline.start();

    auto device = pipeline.getDefaultDevice();

    dai::CalibrationHandler calibration = device->readCalibration();
    std::vector<std::vector<float>> intrinsics = calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, SIZE.width, SIZE.height);

    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    std::cout << "Camera Intrinsics from Sensor:" << std::endl;
    std::cout << "  fx: " << fx << ", fy: " << fy << std::endl;
    std::cout << "  cx: " << cx << ", cy: " << cy << std::endl;

    int argc = 1;
    char* argv[1] = {(char*)"app"};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(g_winW, g_winH);
    glutCreateWindow("OpenGL Point Cloud Viewer");
    glutDisplayFunc(displayGL);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(16, timerFunc, 0);

    std::thread t([syncQueue, fx, fy, cx, cy]()
                  {
        while (true)
        {
            auto messageGroup = syncQueue->get<dai::MessageGroup>();
            if (!messageGroup) continue;
            auto frameRgb = messageGroup->get<dai::ImgFrame>("rgb");
            auto frameDepth = messageGroup->get<dai::ImgFrame>("depth_aligned");
            if (!frameDepth) continue;
            cv::Mat depthMat = frameDepth->getFrame();
            cv::Mat rgbMat;
            if (frameRgb) rgbMat = frameRgb->getCvFrame();

            std::vector<float> pts;
            std::vector<unsigned char> cols;
            buildPointArray(depthMat, rgbMat, fx, fy, cx, cy, pts, cols);

            {
                std::lock_guard<std::mutex> lk(g_mutex);
                g_points.swap(pts);
                g_colors.swap(cols);
            }
            glutPostRedisplay();
        } });
    t.detach();

    glutMainLoop();

    return 0;
}
