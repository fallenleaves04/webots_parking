import numpy as np
import cv2
from controller import (Robot, Keyboard, Supervisor,Display)
from vehicle import (Driver, Car)
import torchvision.transforms as T

import sys
import visualise as vis
import camera_calibration as cc
import fisheye_camera_calibration as fcc
from park_algo import TrajStateMachine,Kalman,Trajectory,rear_wheel_feedback_control,generate_parking_map,wrap_angle,C
import stereo_yolo as sy
from ultralytics import YOLO

import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import math
import pathlib
sys.path.append(r"D:\\User Files\\BACHELOR DIPLOMA\\Kod z Github (różne algorytmy)")


#from PIL import Image
# --------------------- Stałe ---------------------
TIME_STEP = 64
NUM_DIST_SENSORS = 12
NUM_CAMERAS = 8
MAX_SPEED = 250.0
CAMERA_HEIGHT = 2160
CAMERA_WIDTH = 3840

SENSOR_INTERVAL = 0.064
IMAGE_INTERVAL  = 0.2
KEYBOARD_INTERVAL = 0.04

# --------------------- Zmienne globalne ---------------------

robot = Robot()
driver = Car()
supervisor = Supervisor()

display = Display('display')
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

#Driver.synchronization = True
#Robot.synchronization = True
#Supervisor.synchronization = True
#Car.synchronization = True
cameras = []
camera_names = []
cam_matrices = {}
images =[]
front_sensors = []
rear_sensors = []
right_side_sensors = []
left_side_sensors = []

front_sen_apertures = []
rear_sen_apertures = []
right_side_sen_apertures = []
left_side_sen_apertures = []

steering_angle = 0.0
manual_steering = 0

previous_error = 0.0
integral = 0.0
homography_matrices = {}

path_to_models = r"D:\\User Files\\BACHELOR DIPLOMA\\Modele sieci\\"


#DETECTRON2 DLA PANOPTIC ORAZ DEEPLABV3+

"""
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.projects.deeplab import add_deeplab_config


cfg_panoptic = get_cfg()
add_panoptic_deeplab_config(cfg_panoptic)
cfg_panoptic.merge_from_file(r"C:\\Users\\fbiwa\\detectron2\\projects\\Panoptic-DeepLab\\configs\\Cityscapes-PanopticSegmentation\\panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml")
cfg_panoptic.MODEL.WEIGHTS = path_to_models + "panoptic-deeplab.pkl"
cfg_panoptic.MODEL.DEVICE = "cuda"
cfg_panoptic.MODEL.PANOPTIC_DEEPLAB.INSTANCES = True
cfg_panoptic.freeze()


cfg_deeplab = get_cfg()
add_deeplab_config(cfg_deeplab)
cfg_deeplab.merge_from_file(r"C:\\Users\\fbiwa\\detectron2\\projects\\DeepLab\\configs\\Cityscapes-SemanticSegmentation\\deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
cfg_deeplab.MODEL.WEIGHTS = path_to_models + "deeplabv3+.pkl"
cfg_deeplab.MODEL.DEVICE = "cuda"
cfg_deeplab.freeze()

predictor_panoptic = DefaultPredictor(cfg_panoptic)
"""


# --------------------- Helper Functions ---------------------
def print_help():
    print("Samochód teraz jeździ.")
    print("Proszę użyć klawiszy UP/DOWN dla zwiększenia prędkości lub LEFT/RIGHT dla skrętu")
    print("Naciśnij klawisz P, aby rozpocząć poszukiwanie miejsca")
    print("Podczas parkowania, wciśnij Q aby szukać z prawej strony")
    print("albo E aby szukać miejsca z lewej strony")

def set_speed(kmh,driver):
    global speed
    speed = min(kmh, MAX_SPEED)
    driver.setCruisingSpeed(speed)
    print(f"Ustawiono prędkość {speed} km/h")

def set_steering_angle(wheel_angle,driver):
    global steering_angle
    # Clamp steering angle to [-0.5, 0.5] radians (per vehicle constraints)
    wheel_angle = max(min(wheel_angle, C.MAX_WHEEL_ANGLE), -C.MAX_WHEEL_ANGLE)
    steering_angle = wheel_angle
    driver.setSteeringAngle(steering_angle)
    print(f"Skręcam {steering_angle} rad")

def change_manual_steering_angle(inc,driver):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02,driver)



#----------------------Sensor functions-----------------

camera_names = [
        "camera_rear","camera_front_bumper",
        "camera_front_right","camera_front_left",
        "camera_left_mirror","camera_right_mirror",
        "camera_front_right(1)"
    ]

def get_camera_image(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img = camera.getImage()
    if img is None:
        return None

    img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


front_sensor_names = ["distance sensor front right", "distance sensor front righter",
"distance sensor front lefter", "distance sensor front left"
]
rear_sensor_names = ["distance sensor right", "distance sensor righter",
"distance sensor lefter", "distance sensor left"
]
left_side_sensor_names = ["distance sensor left front side","distance sensor left side"]
right_side_sensor_names = ["distance sensor right front side","distance sensor right side"]

def process_distance_sensors(sen):
    l_dist = sen.getLookupTable()
    a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
    b_dist = l_dist[3]-l_dist[4]*a_dist
    value = sen.getValue()
    distance = a_dist*value+b_dist
    sigma = l_dist[2]
    noisy_distance = distance + np.random.normal(0, sigma)
    return noisy_distance


# --------------------- Main Controller Loop ---------------------

speed = 0.0
odom = 0.0
speed = 0.0
spot = 0.0
now = 0.0
prev_x = 0.0
prev_y = 0.0
yaw_init = 0.0
fig, ax_cones, ax_live = None, None, None

parker = None

print("reading homographies...")
right_H = np.load("matrices/right_homo.npy").astype(np.float32)
left_H = np.load("matrices/left_homo.npy").astype(np.float32)
front_H = np.load("matrices/front_homo.npy").astype(np.float32)
right_fender_H = np.load("matrices/right_fender_homo.npy").astype(np.float32)
left_fender_H = np.load("matrices/left_fender_homo.npy").astype(np.float32)
rear_H = np.load("matrices/rear_homo.npy").astype(np.float32)

homographies = []
homographies.extend([front_H,right_H,right_fender_H,
rear_H,left_fender_H,left_H])


s = 2 # skala
S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)
homographies = [S @ H @ np.linalg.inv(S) for H in homographies]

print("reading transformation matrices...")

front_T = np.load("matrices/camera_front_bumper_wide_T_global.npy").astype(np.float32)
left_T = np.load("matrices/camera_left_pillar_T_global.npy").astype(np.float32)
right_T = np.load("matrices/camera_right_pillar_T_global.npy").astype(np.float32)
left_fender_T = np.load("matrices/camera_left_fender_T_global.npy").astype(np.float32)
right_fender_T = np.load("matrices/camera_right_fender_T_global.npy").astype(np.float32)
rear_T = np.load("matrices/camera_rear_T_global.npy").astype(np.float32)
front_right_T = np.load("matrices/camera_front_right_T_global.npy").astype(np.float32)
front_left_T = np.load("matrices/camera_front_left_T_global.npy").astype(np.float32)

stream1 = cv2.cuda.Stream()
stream2 = cv2.cuda.Stream()
stream3 = cv2.cuda.Stream()
stream4 = cv2.cuda.Stream()
stream5 = cv2.cuda.Stream()
stream6 = cv2.cuda.Stream()
stream7 = cv2.cuda.Stream()
stream8 = cv2.cuda.Stream()
stream9 = cv2.cuda.Stream()
stream10 = cv2.cuda.Stream()
stream11 = cv2.cuda.Stream()
stream12 = cv2.cuda.Stream()
stream13 = cv2.cuda.Stream()

streams = (stream1,stream2,stream3,stream4,stream5,
stream6,stream7,stream8,stream9,stream10,stream11,stream12,stream13)

# dla czujników ultradźwiękowych
dists = []
max_min_dict = {}






# dla wizualizacji, pokazywania okien i przesyłania danych
class VisController(vis.QtCore.QObject):

    parkingToggled = vis.QtCore.pyqtSignal(bool)
    sensorUpdated = vis.QtCore.pyqtSignal(object)
    locUpdated = vis.QtCore.pyqtSignal(object)
    trajUpdated = vis.QtCore.pyqtSignal(object)
    speedUpdated = vis.QtCore.pyqtSignal(object)      
    angleUpdated = vis.QtCore.pyqtSignal(object)  

    
    def __init__(self):
        super().__init__()
        self.parking = False
        
    @vis.QtCore.pyqtSlot()
    def toggle_parking(self):
        self.parking = not self.parking
        self.parkingToggled.emit(self.parking)
   

# IMPLEMENTACJA MAIN ALE W QTHREAD, ŻEBY ZROBIĆ WIZUALIZACJĘ DOBRĄ ; CAŁE RUN MOŻNA PRZENIEŚĆ DO DEF MAIN(), JEŻELI QT NIEPOTRZEBNE


class MainWorker(vis.QtCore.QObject):
    
    sensorData = vis.QtCore.pyqtSignal(object)
    poseData   = vis.QtCore.pyqtSignal(object)
    finished   = vis.QtCore.pyqtSignal(bool)
    trajData   = vis.QtCore.pyqtSignal(object)
    speedData  = vis.QtCore.pyqtSignal(object)  
    angleData  = vis.QtCore.pyqtSignal(object)
    def __init__(self,supervisor):
        super().__init__()
        self.first_call_pose = True
        self.first_call_traj = True
        self.node = supervisor.getSelf()
        self.writeParkingPose = False
            
    @vis.QtCore.pyqtSlot()
    def run(self):

        for name in front_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                front_sensors.append(sen)
                l_dist = sen.getLookupTable()
                front_sen_apertures.append(sen.getAperture())
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]

        for name in rear_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                rear_sensors.append(sen)
                l_dist = sen.getLookupTable()
                rear_sen_apertures.append(sen.getAperture())
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]
        for name in left_side_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                left_side_sensors.append(sen)
                l_dist = sen.getLookupTable()
                left_side_sen_apertures.append(sen.getAperture())
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]
        for name in right_side_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                right_side_sensors.append(sen)
                l_dist = sen.getLookupTable()
                right_side_sen_apertures.append(sen.getAperture())
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]

        # Inicjalizuj kamery
        for name in camera_names:
            cam = robot.getDevice(name)
            if cam:
                cam.enable(TIME_STEP)
                cameras.append(cam)
                width = cam.getWidth()
                height = cam.getHeight()
                fov_rad = cam.getFov()
                Kmat = sy.calculate_intrinsic_matrix(width, height, fov_rad)
                cam_matrices[name] = Kmat
                #print(f"Włączono kamerę: {name}")

        # GPS inicjalizacja
        gps = robot.getDevice("gps")
        if gps:
            gps.enable(TIME_STEP)
        #Inicjalizacja IMU
        imu = robot.getDevice("inertial unit")
        if imu:
            imu.enable(TIME_STEP)
        #Inicjalizacja żyroskopu
        gyro = robot.getDevice("gyro")
        if gyro:
            gyro.enable(TIME_STEP)
        #Inicjalizacja akcelerometru
        acc = robot.getDevice("accelerometer")
        if acc:
            acc.enable(TIME_STEP)
        # Wydrukuj wskazówki
        print_help()


        last_sensor_time = robot.getTime()
        last_image_time  = robot.getTime()
        last_key_time = robot.getTime()
        last_meas_time = robot.getTime()

        park_poses = {}
        #now = driver.getTime()

        def check_keyboard(cont):
            global fig, ax_cones
            nonlocal last_key_time
            key = keyboard.getKey()
            #gear = driver.getGear()
            if key == Keyboard.UP:
                set_speed(speed + 0.5,driver)
                # if gear >= 0:
                #     if speed >= 0.0 and speed < 6.0:
                #         driver.setGear(1)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 6.0 and speed < 24.0:
                #         driver.setGear(2)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 24.0 and speed < 40.0:
                #         driver.setGear(3)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 40.0 and speed < 70.0:
                #         driver.setGear(4)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                # elif gear == -1:
                #     driver.setThrottle(0.0)
                #     driver.setBrakeIntensity(1.0)

            elif key == Keyboard.DOWN:
                set_speed(speed - 0.5,driver)

                # if gear == -1:
                #     driver.setThrottle(1.0)
                #     driver.setBrakeIntensity(0.0)
                # elif gear > 0:
                #     driver.setThrottle(0.0)
                #     driver.setBrakeIntensity(1.0)


            elif key == Keyboard.RIGHT:
                change_manual_steering_angle(+5,driver)
            elif key == Keyboard.LEFT:
                change_manual_steering_angle(-5,driver)
            elif key in (ord('p'),ord('P')):

                cont.toggle_parking()
                if cont.parking:

                    print("Rozpoczęto parking")
                else:
                    cv2.destroyAllWindows()
                    print("Ukończono parking")
            

            # elif key in (ord('F'),ord('f')):
            #     driver.setGear(1)
            #     print("Napęd do przodu")
            # elif key in (ord('R'),ord('r')):
            #     driver.setGear(-1)
            #     print("Napęd do tyłu")
            #else:
            #driver.setThrottle(0.0)
            #driver.setBrakeIntensity(0.0)
            
            elif not cont.parking:
                if key==ord('l') or key==ord('L'):
                    if len(park_poses) == 0:
                        print("Brak danych do zapisania.")
                        return
                    path = r"pozy.csv"
                    df = pd.DataFrame([{
                        "x_odo": p["x_odo"],
                        "y_odo": p["y_odo"],
                        "psi_odo": p["psi_odo"],
                        "x_webots": p["node_pos_x"],
                        "y_webots": p["node_pos_y"],
                        "psi_webots": p["yaw_webots"],
                    } for p in park_poses])

                    file_exists = os.path.exists(path)
                    with open(path, mode=("a" if file_exists else "w"), encoding="utf-8", newline="") as f:
                        df.to_csv(f, index=False, header=not file_exists)
                        f.write("\n")

                    print(f"CSV zapisany: {path}, N={len(df)}")
            
            elif cont.parking:
                if key==ord('l') or key==ord('L'):
                    self.writeParkingPose = not self.writeParkingPose
                




        

        #predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-24')
        #painter = KeypointPainter()

        model = YOLO(path_to_models + "yolo11m-seg.pt")


        colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8)

        img_paths = r"D:\\User Files\\BACHELOR DIPLOMA\\test_movie\\"
        i = 0
        prev_time = driver.getTime()
        prev_real = time.time()
        # do supervisora - POZYCJA SAMOCHODU W WEBOTS
        node_pos0 = np.zeros(3)

        # do odometrii


        im0 = [0.0,0.0,0.0]

        gp0 = [0.0,0.0,0.0]
        x_odo = 0.0
        y_odo = 0.0
        sp_odo = 0.0
        psi = 0.0
        delta = 0.0
        yaw_est = 0.0
        yaw_real = 0.0



        # parametry samochodu
        front_radius = driver.getFrontWheelRadius()
        rear_radius = driver.getRearWheelRadius()
        wheelbase = driver.getWheelbase()
        #track_front = driver.getTrackFront()
        #track_rear = driver.getTrackRear()
        #car_type = driver.getType()
        # dla enkoderów i prędkości z odometrii
        encoders = np.zeros(4)
        enc0 = np.zeros(4)
        wheel_speeds = np.zeros(4)
        # przeszłe enkodery
        prev_enc = np.zeros(4)

        

        

        def get_speed_odo(dt):

            # liczymy enkodery
            for i in range(4):
                encoders[i] = driver.getWheelEncoder(i) - enc0[i]
                wheel_speeds[i] = (encoders[i] - prev_enc[i]) / dt
                prev_enc[i] = encoders[i]

            #speed = 0.5 * (wheel_speeds[0] + wheel_speeds[1]) * front_radius
            # speed = 0.25 * (wheel_speeds[0] + wheel_speeds[1]) * front_radius + \
            #         0.25 * (wheel_speeds[2] + wheel_speeds[3]) * rear_radius
            speed = 0.5 * (wheel_speeds[2] + wheel_speeds[3]) * rear_radius

            return speed,encoders

        def webots_to_odom_xy(dx,dy,yaw0):
            c = np.cos(yaw0); s = np.sin(yaw0)
            # obrót o -yaw0
            x_loc =  c*dx + s*dy
            y_loc = -s*dx + c*dy
            return x_loc, y_loc

        def R_xyz(yaw, pitch, roll):
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            return np.array([
                [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                [-sp,    cp*sr,             cp*cr]
            ])
        
        
        def get_pose_kalman(dt,kalman):
            
            nonlocal yaw_est,x_odo,y_odo,sp_odo,delta,yaw_real,gp0,im0,node_pos0,enc0
            if self.first_call_pose:
                x_odo = 0.0
                y_odo = 0.0
                gp0 = gps.getValues()
                im0 = imu.getRollPitchYaw()
                for i in range(4):
                    enc0[i] = driver.getWheelEncoder(i)
                    prev_enc[i] = 0.0
                node_pos0 = self.node.getPosition()
                
                self.first_call_pose = False
                yaw_real = 0.0
            # gps
            node_pos = self.node.getPosition()
            node_vel = self.node.getVelocity()
            # imu
            rpy = imu.getRollPitchYaw()
            yaw_imu = wrap_angle(rpy[2] - im0[2])
            im = [rpy[0] - im0[0], rpy[1] - im0[1], yaw_imu]
            # supervisor
            R = R_xyz(rpy[2], rpy[1], rpy[0])   # ZYX
            node_vel_xyz = R.T @ np.array(node_vel[:3])
            node_vel_x = node_vel_xyz[0]  
            # żyroskop
            gyr = gyro.getValues()
            yaw_est = wrap_angle(yaw_est + gyr[2] * dt)
            # odometria
            sp_odo_meas,encoders = get_speed_odo(dt)
            # kąt skrętu kół [rad]
            delta_meas = -driver.getSteeringAngle()
            x_pred = kalman.predict(np.array([x_odo,y_odo,yaw_real,sp_odo_meas,delta_meas]),dt)
            x_upd = kalman.update(x_pred,np.array([yaw_est]))
            x_odo = x_upd[0]
            y_odo = x_upd[1]
            yaw_real = x_upd[2]
            sp_odo = x_upd[3]
            delta = x_upd[4]
            # akcelerometr
            accer = acc.getValues()
            # node position
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]
            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":yaw_real,"dt":dt,"x_odo":x_odo,"y_odo":y_odo,"encoders":encoders,"node_pos":node_pos,"acc":accer,"node_vel":node_vel_xyz,"node_vel_x":node_vel_x}

        R0 = np.eye(3)

        def get_pose(dt):
            
            nonlocal psi,x_odo,y_odo,gp0,im0,node_pos0,yaw_est,R0
            # nadanie wartości początkowych
            if self.first_call_pose:
                x_odo = 0.0
                y_odo = 0.0
                gp0 = gps.getValues()
                im0 = imu.getRollPitchYaw()
                for i in range(4):
                    enc0[i] = driver.getWheelEncoder(i)
                    prev_enc[i] = 0.0
                node_pos0 = self.node.getPosition()
                R0 = R_xyz(im0[2], im0[1], im0[0])   # orientacja początkowa
                self.first_call_pose = False
                psi = 0.0
            # supervisor
            node_pos = self.node.getPosition()
            node_vel = self.node.getVelocity()
                                                                  
            
            # imu
            rpy = imu.getRollPitchYaw()
            yaw_imu = wrap_angle(rpy[2] - im0[2])
            im = [wrap_angle(rpy[0] - im0[0]), wrap_angle(rpy[1] - im0[1]), yaw_imu]
            
            
            R = R_xyz(rpy[2], rpy[1], rpy[0])   # ZYX
            
            node_vel_xyz = R.T @ np.array(node_vel[:3])
            node_vel_x = node_vel_xyz[0]  
            # żyroskop
            gyr = gyro.getValues()
            #odometria, przednie koła
            sp_odo,encoders = get_speed_odo(dt)
            delta = -driver.getSteeringAngle()  # kąt skrętu kół [rad]
            psi = psi + (sp_odo * np.tan(delta) / wheelbase) * dt

            #yaw_real = wrap_angle(0.995*yaw_est + 0.005*psi)
            yaw_real = wrap_angle(psi)
            
            x_odo += sp_odo * dt * np.cos(yaw_real)
            y_odo += sp_odo * dt * np.sin(yaw_real)
            # supervisor
            
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]
            
            # akcelerometr
            accer = acc.getValues()

            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":yaw_real,"dt":dt,"x_odo":x_odo,
                    "y_odo":y_odo,"encoders":encoders,"node_pos":node_pos,"acc":accer,
                    "node_vel":node_vel_xyz,"node_vel_x":node_vel_x}

        # do wyswietlania punktów z YOLO
        Mtr = np.eye(4)
        Mtr[0,3] = -C.CAR_LENGTH + 1
        name = "camera_front_right"
        T_center_to_front = Mtr @ front_right_T 
        ptt = sy.project_points_world_to_image(np.array([[0.0, 0.0, 0.0]], dtype=np.float32),T_center_to_front,cam_matrices[name])
        # obwód koła
        circ = 2*np.pi*(front_radius+rear_radius)*0.5
        #circ *= 0.5 
        #print(f"Obwód koła: {circ}")

        kalman = Kalman(wheelbase)
        v_kmh = 4.0      
        tsm = TrajStateMachine(driver,self)  
    

        #hastar = HybridAStar()
        map_cars = generate_parking_map()
        name = "camera_front_right"
        pattern_size = (4,3)
        
        import reeds_shepp

        start = (0.0, 0.0, 0.0)    # x, y, yaw [rad]
        goal  = (20.0, 15.0, np.pi)
        radius = wheelbase/np.tan(0.5)               # promień skrętu (m)
        l1 = reeds_shepp.path_length(start, goal, radius)
        sl1 = reeds_shepp.path_type(start, goal, radius)
        segments,lengths = sl1[0],sl1[1]
        path = reeds_shepp.path_sample(start, goal, radius, step_size=0.05)

        start = goal
        goal = (0.0,40.0,np.pi/4)
        path2 = reeds_shepp.path_sample(start,goal,radius,step_size=0.05)
        sl2= reeds_shepp.path_type(start, goal, radius)
        segments2,lengths2 = sl2[0],sl2[1]
        l2 = reeds_shepp.path_length(start, goal, radius)
        path += path2
        sref = np.zeros(len(path))

        
        curv_dict = {"L":C.MAX_CURVATURE,"R":-C.MAX_CURVATURE,"S":0.0}
        s_l = 0

        segments += segments2
        lengths += lengths2
        cx,cy,cyaw,ccurv = [],[],[],[]

        
        speed_ref = 0.0
        ref_path = Trajectory(cx, cy, cyaw,ccurv)

        # STEREOWIZJA ____________________
        name_right = "camera_front_right"
        name_left = "camera_front_left"
        # Obiekt stereo matcher
        stereo_left = cv2.cuda.StereoSGM.create(minDisparity=0,
        numDisparities=32,
        blockSize=20,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=8)
        # matcher dla prawego obrazu - trzeba użyć createRightMatcher z ximgproc
        stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
        stereo_right = cv2.cuda.StereoSGM.create(minDisparity=0,
        numDisparities=32,
        blockSize=20,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=8)
        # utwórz filtr WLS
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
        wls_filter.setLambda(4000)
        wls_filter.setSigmaColor(2.5)
        K_right = cam_matrices[name_right]
        K_left = cam_matrices[name_left]
        f_left = K_left[0][0]
        f_right = K_right[0][0]


        while robot.step(96) != -1:
            pts = []

            now = driver.getTime()
            dt_sim = now - prev_time
            
            now_real = time.time()
            dt_real = now_real - prev_real
            
            prev_real = now_real
            prev_time = now
            
            #print(f"Pozycja z Webots {node.getPosition()}")
            #print(f"Pedał wciśnięty: {driver.getThrottle()}")
            names_images = dict(zip(camera_names, [get_camera_image(c) for c in cameras]))
            
            
            if cont.parking:
                
                pose_measurements = get_pose(dt_sim)
                
                x_odo = pose_measurements["x_odo"]
                y_odo = pose_measurements["y_odo"]
                yaw_odo = pose_measurements["psi"]
                delta_meas = pose_measurements["delta"]
                sp_odo = pose_measurements["sp_odo"]
                node_vel_x = pose_measurements["node_vel_x"]
                node_pos = pose_measurements["node_pos"]
                yaw_webots = pose_measurements["im"][2]
                
                if self.writeParkingPose:
                    self.writeParkingPose = not self.writeParkingPose
                    #park_poses.update({"x_odo":x_odo,"y_odo":y_odo,"node_pos_x":node_pos[0],"node_pos_y":node_pos[1],"psi_odo":yaw_odo,"psi_webots":yaw_webots})
                    #print("Zapisano miejsce parkingowe")
                
                front_dists = [process_distance_sensors(s) for s in front_sensors]
                rear_dists = [process_distance_sensors(s) for s in rear_sensors]
                right_side_dists = [process_distance_sensors(s) for s in right_side_sensors]
                left_side_dists = [process_distance_sensors(s) for s in left_side_sensors]

                front_names_dists = dict(zip(front_sensor_names, front_dists))
                rear_names_dists = dict(zip(rear_sensor_names, rear_dists))
                left_side_names_dists = dict(zip(left_side_sensor_names, left_side_dists))
                right_side_names_dists = dict(zip(right_side_sensor_names, right_side_dists))
                
                
                traj_data = [[x_odo,y_odo,yaw_odo],node_pos,map_cars,ref_path]  
                speed_data = [now,sp_odo,node_vel_x]
                angle_data = [now,delta_meas,psi,yaw_webots]
                self.sensorData.emit([front_names_dists,rear_names_dists,
                                      left_side_names_dists,right_side_names_dists,
                                      max_min_dict])
                self.poseData.emit(pose_measurements)
                self.trajData.emit(traj_data)
                self.speedData.emit(speed_data)
                self.angleData.emit(angle_data)
                
                #fcc.solve_camera_pose(image,pattern_size,cam_matrices[name],name)
                image = names_images[name].copy()
                results = model(image,half=True,device = 0,conf=0.6,verbose=False,imgsz=(1280,960))
                
                for box in results[0].boxes.xyxy.cpu().numpy():  # [x1,y1,x2,y2]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                    x,y = (x1+x2)/2,y2
                    pt = sy.pixel_to_world(x,y,cam_matrices[name],T_center_to_camera=T_center_to_front)
                    pts.append(pt)
                    radius = 3
                    color = (0,0,255)  
                    cv2.circle(image, (int(x),int(y)), radius, color, -1) 
                    (u1,v1) = ptt[0] 
                    image = cv2.line(image, (int(x),int(y)), (u1,v1), (0,0,255), 3)  # X
                print(f"Punkty odnalezione z YOLO: {pts}")
                cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
                cv2.imshow("yolo", image)
                
                
                img_right = names_images[name_right]
                img_left = names_images[name_left]
                right_copy = img_right.copy()
                if results and results[0].masks is not None:
                    grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                    grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)


                    #TUTAJ DALEJ ODFILTROWANE DISPARITY
                    # oblicz disparity z lewej i prawej kamery
                    disp_left = stereo_left.compute(grayL, grayR).astype(np.float32) / 16.0
                    disp_right = stereo_right.compute(grayR, grayL).astype(np.float32) / 16.0

                    # filtruj disparity
                    filtered_disp = wls_filter.filter(disp_left, grayL, disp_right, grayR)

                    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
                    disp_vis = np.nan_to_num(disp_vis, nan=0.0, posinf=0.0, neginf=0.0)
                    disp_vis = np.uint8(disp_vis)
                    cv2.namedWindow("Disparity WLS filtered",cv2.WINDOW_NORMAL)
                    cv2.imshow("Disparity WLS filtered", disp_vis)
                                   
                cv2.waitKey(1)
            elif not cont.parking:
                self.first_call_pose = True
                self.first_call_traj = True
                prev_time = driver.getTime()
                prev_real = time.time()
                
            #if now - last_key_time >= KEYBOARD_INTERVAL:
            check_keyboard(cont)
            last_key_time = now
            

        app.quit()

        self.finished.emit()


if __name__ == "__main__":
    app = vis.pg.QtWidgets.QApplication(sys.argv)
    cont = VisController()
    win  = vis.SensorView(cont)
    win.hide()

    win1 = vis.SpeedView(cont)
    win1.hide()

    win2 = vis.AngleView(cont)
    win2.hide()

    win3 = vis.TrajView(cont)
    win3.hide()
    thread = vis.QtCore.QThread()
    worker = MainWorker(supervisor)
    worker.moveToThread(thread)

    # sygnały z mainworker
    thread.started.connect(worker.run)
    worker.sensorData.connect(cont.sensorUpdated)
    worker.poseData.connect(cont.locUpdated)
    worker.trajData.connect(cont.trajUpdated)
    worker.speedData.connect(cont.speedUpdated)
    worker.angleData.connect(cont.angleUpdated)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    sys.exit(app.exec_())


#results = model(names_images[name],half=True,device = 0,conf=0.6)

#annotated_frame = results[0].plot()

#cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
#cv2.imshow("yolo", annotated_frame)
#cv2.waitKey(1)

# Wyniki:
# out["panoptic_seg"] -> (panoptic_map[H,W] (int32), segments_info[list])
# out["sem_seg"]      -> [C,H,W] (logity)
# out["instances"]    -> obiekty (jeśli włączone)

# img = names_images[name]  # HxWx3 (RGB)
# out = predictor_panoptic(img)
# panoptic, segments_info = out["panoptic_seg"]
# panoptic = panoptic.to("cpu").numpy()
# cv2.namedWindow("panoptic", cv2.WINDOW_NORMAL)
# cv2.imshow("panoptic", panoptic)

# Define the codec and create VideoWriter object

"""
if first_call:
    print("Zapis filmiku z kamery.")
    first_call = False
cv2.imwrite(img_paths + f"img_{i}.png",cv2.cvtColor(names_images[name],cv2.COLOR_BGR2RGB))
"""


"""
input_image = names_images[name]
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model_deeplab.to('cuda')
with torch.no_grad():
    output = eff_ps(input_batch)['out'][0]
# Wyniki z modelu
output_predictions = output.argmax(0)

"""


"""
predictions, _, _ = predictor.numpy_image(names_images[name])



out = names_images[name].copy()
for ann in predictions:
    kps = ann.data
    skeleton = ann.skeleton
    #print(ann)
    # rysuj krawędzie
    for (s,e) in skeleton:
        x1,y1,v1 = kps[s-1]
        x2,y2,v2 = kps[e-1]
        if v1 > 0 and v2 > 0:

            thickness = 2
            col = (0,255,0)
            if v1 > 0.5 and v2 > 0.5:
                cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), col, thickness)
            else:
                # przerywana: narysuj kilka krótkich segmentów
                n = 8
                pts = np.linspace([x1,y1],[x2,y2],n)
                for p,q in zip(pts[:-1], pts[1:]):
                    cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), col, 1)

    # rysuj punkty
    for (x,y,v) in kps:
        if v > 0:
            radius = 3
            color = (0,0,255)            # czerwone kropki
            cv2.circle(out, (int(x),int(y)), radius, color, -1)
out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
cv2.imshow('Vehicle Pose', out)

cv2.waitKey(1)
"""






#vis.alt_collect_homo(names_images, homographies, car, streams)
"""
images = [get_camera_image(c) for c in cameras]
names_images = dict(zip(camera_names, images))
name = "camera_front_top"
#sens_queue.put(dists)

results = model(names_images[name])[0]
img = names_images[name].copy()
for result in results:
    bbox = result.boxes.xyxy[0].tolist()

    for kpt in result.keypoints.data[0].tolist():
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

# Display the image with keypoints
cv2.namedWindow("yolo",cv2.WINDOW_NORMAL)
cv2.imshow("yolo",img)
"""
"""
# 2) inference
predictions, _, _ = predictor.numpy_image(names_images[name])

 # 4) rysowanie w OpenCV
out = names_images[name].copy()
for ann in predictions:
    kps = ann.data                      # shape (K,3): x,y,confidence
    skeleton = ann.skeleton             # lista par 1-based

    # rysuj krawędzie
    for (s,e) in skeleton:
        x1,y1,v1 = kps[s-1]
        x2,y2,v2 = kps[e-1]
        if v1 > 0 and v2 > 0:
            # solid jeśli oba > threshold, inaczej przerywana
            thickness = 2
            col = (0,255,0)             # tu możesz wymyślić paletę
            if v1 > 0.5 and v2 > 0.5:
                cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), col, thickness)
            else:
                # przerywana: narysuj kilka krótkich segmentów
                n = 8
                pts = np.linspace([x1,y1],[x2,y2],n)
                for p,q in zip(pts[:-1], pts[1:]):
                    cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), col, 1)

    # rysuj punkty
    for (x,y,v) in kps:
        if v > 0:
            radius = 3
            color = (0,0,255)            # czerwone kropki
            cv2.circle(out, (int(x),int(y)), radius, color, -1)
out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
cv2.imshow('Vehicle Pose', out)

"""
"""
if first_call:
    yaw_init = imu.getRollPitchYaw()[2]
    first_call = False
# automaty parkowania
yaw = imu.getRollPitchYaw()[2] - yaw_init
"""
