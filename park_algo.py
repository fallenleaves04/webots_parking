import math
import numpy as np
import time

import heapq
from heapdict import heapdict


"""
Parametry samochodu
"""
# Vehicle parameters
class C:
    # parametry samochodu 
    TRACK_FRONT = 1.628
    TRACK_REAR = 1.628
    WHEELBASE = 2.995
    MAX_WHEEL_ANGLE = 0.5  # rad
    CAR_WIDTH = 1.95
    CAR_LENGTH = 4.85
    MAX_SPEED = -3.0
    MAX_RADIUS = WHEELBASE/np.tan(MAX_WHEEL_ANGLE)
    MAX_CURVATURE = 1/MAX_RADIUS
    # parametry dla A*
    BACKWARD_COST = 5.0
    GEAR_CHANGE_COST = 5.0
    STEER_CHANGE_COST = 1.0
    STEER_ANGLE_COST = 1.0
    H_COST = 15.0
    # parametry dla próbkowania
    XY_RESOLUTION = 0.05 # m
    YAW_RESOLUTION = np.deg2rad(10.0)
    # parametry dla Stanley
    K_STANLEY = 0.5

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

"""
DLA STANLEY, SKOPIOWANE Z https://github.com/zhm-real/MotionPlanning I DOPASOWANE
"""

        
class Trajectory:
    def __init__(self, cx, cy, cyaw,ccurv):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ccurv = ccurv
        self.ind_old = 0

        self.int_v = 0.0
        self.kp = 0.1
        self.ki = 0.6
        self.kd = 0.001

        self.len = len(self.cx)
        self.s0 = 1

    def calc_theta_e_and_er(self, x,y,yaw):
        """
        calc theta_e and er.
        theta_e = theta_car - theta_path
        er = lateral distance in frenet frame

        :param node: current information of vehicle
        :return: theta_e and er
        """

        ind = self.nearest_index(x,y)

        k = self.ccurv[ind]
        yaw = self.cyaw[ind]

        rear_axle_vec_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                         [math.sin(yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[x - self.cx[ind]],
                                      [y - self.cy[ind]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        theta_e = wrap_angle(yaw- self.cyaw[ind])

        return theta_e, er, k, yaw, ind

    def nearest_index(self, xx,yy):
        """
        find the index of the nearest point to current position.
        :param node: current information
        :return: nearest index
        """

        dx = [xx - x for x in self.cx]
        dy = [yy - y for y in self.cy]
        dist = np.hypot(dx, dy)
        self.s0 += np.argmin(dist[self.s0:self.len])

        return self.s0
    
    def pid_control(self,target_v, v, dist,dt):
        """
        PID controller and design speed profile.
        :param target_v: target speed
        :param v: current speed
        :param dist: distance to end point
        :return: acceleration
        """
        
        
        if dist < 10.0:
            if v > 3.0:
                a = -2.5
            elif v < -2.0:
                a = -1.0
        else:
            err = target_v - v
            diff_v = err/dt
            self.int_v += err*dt 
            a = self.kp*err + self.ki*self.int_v + self.kd*diff_v
            
        return a

def rear_wheel_feedback_control(x,y,v,yaw, ref_path):
    """
    rear wheel feedback controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :return: optimal steering angle
    """

    theta_e, er, k, yaw, ind = ref_path.calc_theta_e_and_er(x,y,yaw)
    

    omega = v * k * math.cos(theta_e) / (1.0 - k * er) - \
            C.K_theta * abs(v) * theta_e - C.K_e * v * math.sin(theta_e) * er / theta_e

    delta = math.atan2(C.WB * omega, v)

    return delta, ind


"""
DALEJ MOJE!________________________________________________________________________________________________________________________________________________________________________________________________!
"""

"""
Funkcje odpowiadające za tworzenie trajektorii Hybrid-A* z rozszerzeniami o ciągłej krzywiźnie,
wraz z tworzeniem mapy przeszkód
"""
    
def generate_parking_map(x_start=0.0, y_start = 4.0, # gdzie zacząć mapę
                            n_c=10, # liczba samochodów
                            l_c = 5,l_p = 2*4.85, # długość samochodów/miejsca parkingowego
                            w_c = 1.9): # szerokość samochodów
    cars = []
    x_center = x_start
    for i in range(n_c): # liczba samochodów
        dist = np.random.uniform(low = 0.2,high = l_p)
        x_center += dist + l_c/2
        yaw = np.random.uniform(low=-0.05,high=0.05)
        y_start += np.random.uniform(low=-0.3,high=0.3)
        cars.append((x_center, y_start,l_c, w_c,yaw))  # 1 pas samochodów z losową odległością od siebie i orientacją
        x_center += l_c/2
    return cars

class OccupancyGrid():
    """
    Siatka zajętości, stosująca zarówno kamery z YOLO, jak i odwrotny model czujnika ultradźwiękowego (ISM) dla mapowania przestrzeni
    """
    def __init__(self,sensors,cameras):
        self.size = (10.0,10.0)
        self.XY_RESOLUTION = C.XY_RESOLUTION
        pass

    def calculate(self,yolo_points,sensor_readings):

        pass

class Path:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.yaws = []
        self.indices = []
        self.curvs = []
        self.g_cost = 0.0
        self.h_cost = 0.0
        self.f_cost = self.g_cost + self.h_cost
        self.prev_ind = 0.0
    def check_car_collision(self):
        """
        Funkcja sprawdzająca kolizję samochodu z przeszkodami (przecięcie narożników karoserii z
        geometriami, liniami itd.)
        """
        pass

    def analytic_expansion(self):
        """
        Funkcja obliczająca rozszerzenia node'ów o krzywe i licząca koszty
        """
        pass

    def update_node_analytic_expansion(self):
        pass

    
    def hybrid_a_star_planning(self,start_pose,goal_pose,occupancy_grid):
        open_set = []
        closed_set = []
        open_set.append(start_pose)
        closed_set.append(goal_pose.index)

        while open_set:
            break
        pass

class Kalman():
    def __init__(self,wheelbase):
        self.states = 5

        self.Q = np.diag([1e-6,1e-6,1e-6,1e-12,1e-12]) # macierz kowariancji szumu procesowego
        self.R = np.diag([(1e-8)**2]) # kowariancji szumu pomiarowego
        self.H = np.array([[0,0,1,0,0]]) # macierz obserwacji
        self.E = np.eye(self.states) # macierz kowariancji błędu
        self.I = np.eye(self.states)
        self.wheelbase = wheelbase
        
        # szumy procesowe i pomiarowe
    
    def compute_F(self,x, dt):
        F = np.eye(self.states)
        v = x[3]
        psi = x[2]
        delta = x[4]
        # rząd 1 x
        # F[0,2] = -dt*np.sin(psi)*np.cos(delta)*v
        # F[0,3] = dt*np.cos(psi)*np.cos(delta)
        # F[0,4] = -dt*np.cos(psi)*np.sin(delta)*v
        F[0,2] = -dt*np.sin(psi)*v
        F[0,3] = dt*np.cos(psi)
        # rząd 2 y
        # F[1,2] = dt*np.cos(psi)*np.cos(delta)*v
        # F[1,3] = dt*np.sin(psi)*np.cos(delta)
        # F[1,4] = -dt*np.sin(psi)*np.sin(delta)*v
        F[1,2] = dt*np.cos(psi)*v
        F[1,3] = dt*np.sin(psi)
        # rząd 3 psi
        # F[2,3] = 1/self.wheelbase*np.sin(delta)*dt
        # F[2,4] = v/self.wheelbase*np.cos(delta)*dt
        F[2,3] = 1/self.wheelbase*np.tan(delta)*dt
        F[2,4] = v/self.wheelbase*(1/(np.cos(delta))**2)*dt
        # rząd 4 i 5 (v i delta) nie tykamy
        return F
    def f(self, x, dt):
        # z przednich kół
        x_model = x[0]
        y_model = x[1]
        psi_model = x[2]
        v_model = x[3]
        delta_model = x[4]
        #x_model = x_model + dt * v_model * np.cos(psi_model) * np.cos(delta_model)
        x_model = x_model + dt * v_model * np.cos(psi_model)
        #y_model = y_model + dt * v_model * np.sin(psi_model) * np.cos(delta_model)
        y_model = y_model + dt * v_model * np.sin(psi_model)
        #psi_model = wrap_angle(psi_model + dt * (v_model/self.wheelbase) * np.sin(delta_model))
        psi_model = wrap_angle(psi_model + dt * (v_model/self.wheelbase) * np.tan(delta_model))
        return np.array([x_model, y_model, psi_model, v_model, delta_model])
    def predict(self,x,dt):
        x_pred = self.f(x,dt)
        F = self.compute_F(x,dt)
        self.E = F @ self.E @ F.T + self.Q
        return x_pred
    def update(self,x_hat,z):
        # z = [psi_meas, vF_meas, delta_meas]
        zhat = np.array([x_hat[2]])
        innov = z - zhat
        innov[0] = wrap_angle(innov[0]) # obetnij yaw, żeby był maks. 2pi
        S = self.H @ self.E @ self.H.T + self.R     # 3x3
        K = self.E @ self.H.T @ np.linalg.inv(S)    # 5x3
        x_upd = x_hat + K @ innov
        x_upd[2] = wrap_angle(x_upd[2])
        I_KH = self.I - K @ self.H
        self.E = I_KH @ self.E @ I_KH.T + K @ self.R @ K.T
        self.x = x_upd
        return x_upd

        
class TrajStateMachine():
    def __init__(self,driver,worker):
        # do symulacji
        self.worker = worker
        self.driver = driver
        # do pomiarów
        self.straight_phase_time = 5.0
        self.curve_phase_time = 5.0
        self.prev_time = 0.0
        self.target_steer = 0.0
        self.target_speed = 0.0
        self.max_angle = 0.5
        self.deriv = 0.0
        self.prev_heading = 0.0
        self.delta_rate = 0.6
        self.speed_rate = 7.0
        self.state_index = 0
        # do symulacji kąta
        self.input = 0.0
        self.u_old = 0.0
        self.u_old_old = 0.0
        self.y = 0.0
        self.y_old = 0.0
        self.y_old_old = 0.0
        self.steer_cmd = 0.0
        # do symulacji prędkości
        self.input_s = 0.0
        self.u_old_s = 0.0
        self.u_old_old_s = 0.0
        self.y_s = 0.0
        self.y_old_s = 0.0
        self.y_old_old_s = 0.0
        self.speed_cmd = 0.0
        self.sequence = [
            ("prosto",        6.0,  8.0,  0.0),
            ("hamowanie",      5.0,  0.0,  0.0),
            ("wstecz_rozp",    3.0, -4.0,  0.0),
            ("wstecz_lewo",    5.0, -4.0, -0.5),
            ("wstecz_prawo",   7.5, -4.0,  0.5),
            ("wstecz_prosto",  4.0, -4.0,  0.0),
            ("hamowanie_stop", 3.0,  0.0,  0.0),
        ]

        
    def steer(self,dt):
        
        self.input = self.target_steer
        
        ts = dt
        tc = 1.0
        e = 0.4 
        k = 1.0
        w = 2 * np.pi / tc 
        a=k*w*ts*ts/(4+4*e*w*ts+ts*ts*w*w)
        b=(ts*ts*w*w/2-2)/(1+e*w*ts+ts*ts/4*w*w)
        c=(1-e*w*ts+ts*ts/4*w*w)/(1+e*w*ts+ts*ts/4*w*w)
        g0 = 4 * a / (1 + b + c)
        if g0 != 0.0:
            a = a / g0
        self.y = a*self.input+ 2*a*self.u_old + a*self.u_old_old - b*self.y_old - c*self.y_old_old
        max_rate = self.delta_rate  # [rad/s] – dostosuj do swojego modelu
        dy = np.clip(self.y - self.steer_cmd, -max_rate * dt, max_rate * dt)
        self.steer_cmd += dy

        # fizyczny limit kąta
        self.steer_cmd = np.clip(self.steer_cmd, -self.max_angle, self.max_angle)

        # przesunięcie stanów
        self.u_old_old = self.u_old
        self.u_old = self.input
        self.y_old_old = self.y_old
        self.y_old = self.y
        
        self.driver.setSteeringAngle(self.steer_cmd)
    def cont_speed(self,dt):
        self.input_s = self.target_speed
        
        ts = dt
        tc = 0.1
        e = 0.9
        k = 1.0
        w = 2 * np.pi / tc 
        a=k*w*ts*ts/(4+4*e*w*ts+ts*ts*w*w)
        b=(ts*ts*w*w/2-2)/(1+e*w*ts+ts*ts/4*w*w)
        c=(1-e*w*ts+ts*ts/4*w*w)/(1+e*w*ts+ts*ts/4*w*w)
        g0 = 4 * a / (1 + b + c)
        if g0 != 0.0:
            a = a / g0
        self.y_s = a*self.input_s + 2*a*self.u_old_s + a*self.u_old_old_s - b*self.y_old_s - c*self.y_old_old_s
        max_rate = self.speed_rate 
        dy = np.clip(self.y_s - self.speed_cmd, -max_rate * dt, max_rate * dt)
        self.speed_cmd += dy

        # przesunięcie stanów
        self.u_old_old_s = self.u_old_s
        self.u_old_s = self.input_s
        self.y_old_old_s = self.y_old_s
        self.y_old_s = self.y_s
        if self.target_speed == 0.0 and abs(self.speed_cmd - self.target_speed) < 1e-3:
            self.driver.setCruisingSpeed(self.target_speed)
        else:
            self.driver.setCruisingSpeed(self.speed_cmd)

    def update(self,now,dt):
        """
        if self.state_index >= len(self.sequence):
            # koniec programu
            self.target_speed = 0.0
            self.target_steer = 0.0
            self.steer(dt)
            self.cont_speed(dt)
        
        """
        if self.worker.first_call_traj:
            self.prev_time = time.time()
            self.worker.first_call_traj = False
            
        state_name, duration, self.target_speed, self.target_steer = self.sequence[self.state_index]
            

        # przejście do następnego stanu po upływie czasu
        if now - self.prev_time >= duration:
            self.state_index += 1
            self.prev_time = now
            if self.state_index < len(self.sequence):
                next_state = self.sequence[self.state_index][0]
                print(f"{now:6.2f}s -> zmiana: {state_name} - {next_state}")
            else:
                self.state_index = 0

        # wykonanie dynamiki
        self.steer(dt)
        self.cont_speed(dt)


class Parking:
    # Domyślne marginesy startu i końca manewru
    marg_start = 0.2
    marg_end   = 0.4


    def __init__(self, driver, side, times,
                 min_width=C.CAR_WIDTH*1.1,
                 min_length=C.CAR_LENGTH*1.25,
                 threshold=1*C.CAR_WIDTH):
        # Inicjalizacja parametrów parkingu
        self.min_width  = min_width       # minimalna szerokość miejsca
        self.min_length = min_length      # minimalna długość miejsca
        self.threshold  = threshold       # próg zmiany odległości

        self.state           = "searching_start"  # aktualny stan maszyny stanów
        self.start_pose      = None               # miejsce wykrycia początku luki
        self.spots           = []                 # wykryte miejsca parkingowe
        self.driver          = driver             # interfejs do samochodu w Webots
        self.spot = None
        # Pomocnicze zmienne do detekcji zmian odległości
        self.prev_distance_front = 6.0
        self.prev_distance_rear  = 6.0
        self.dist_start_far      = 6.0
        self.dist_start_cl       = 6.0
        self.dist_end_far        = 6.0
        self.dist_end_cl         = 6.0

        self.side      = side    # strona parkowania: "left" lub "right"
        self.x = 0.0             # os x w układzie globalnym
        self.y = 0.0             # os y w układzie globalnym
        self.yaw = 0.0           # orientacja pojazdu
        self.last_time = times   # czas ostatniej aktualizacji

        # Parametry regulatora PID dla sterowania
        self.Kp = 1.2
        self.Kd = 0.001
        self.Kl = 0.5
        self.prev_yaw_err = 0.0

    def update_odometry(self, yaw):
        """
        Aktualizuje pozycję (x,y) na podstawie prędkości i kąta yaw.
        """
        # Pobierz prędkość w km/h i przelicz na m/s
        v = self.driver.getCurrentSpeed() / 3.6

        # Integracja prostokątna z krokiem czasowym 0.05s
        self.yaw = yaw
        dx = v * math.cos(self.yaw) * 0.06
        dy = v * math.sin(self.yaw) * 0.06
        self.x += dx
        self.y += dy

        return (self.x, self.y, self.yaw)

    def update_state(self, dists_names, yaw):
        """
        Maszyna stanów wykrywająca luki parkingowe.
        """
        # Wybór odpowiednich czujników w zależności od strony
        if self.side == "left":
            distance_front = dists_names["distance sensor left front side"]
            distance_rear  = dists_names["distance sensor left side"]
        else:  # self.side == "right"
            distance_front = dists_names["distance sensor right front side"]
            distance_rear  = dists_names["distance sensor right side"]

        # Oblicz zmiany odległości
        delta_front = distance_front - self.prev_distance_front
        delta_rear  = distance_rear  - self.prev_distance_rear
        self.prev_distance_front = distance_front
        self.prev_distance_rear  = distance_rear

        # Aktualizuj pozycję z odometrii
        odom_pose = self.update_odometry(yaw)
        x, y, yaw = odom_pose
        #print(f"Odometria : {odom_pose}")
        # Stan: poszukiwanie początku luki
        if self.state == "searching_start":
            if delta_front > self.threshold:
                # Znaleziono gwałtowny wzrost odległości ⇒ początek luki
                self.start_pose      = (x - self.marg_start, y, yaw)
                self.dist_start_far  = distance_front
                self.dist_start_cl   = self.prev_distance_front
                self.state           = "searching_progress"
                print("Kandydat na miejsce znaleziony.")

        # Stan: poszukiwanie końca luki
        elif self.state == "searching_progress":

            if -delta_front > self.threshold:
                # Znaleziono gwałtowny spadek odległości ⇒ koniec luki
                end_pose = (x + self.marg_end, y, yaw)
                self.dist_end_far = distance_front
                self.dist_end_cl  = distance_front - delta_front

                # Utwórz obiekt miejsca parkingowego
                spot = self._make_spot(self.start_pose, end_pose)
                if spot:
                    self.state = "waiting_for_park"
                    self.spot = spot
                    print("Miejsce znalezione. Wciśnij Y, aby rozpocząć parkowanie. (NIE ZALECANE, NIE UMIE PARKOWAĆ)", spot)
                else:
                    print("Miejsce okazało się za małe.")
                    self.state = "searching_start"

        if self.spot is not None: return odom_pose, self.spot

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2*math.pi
        while angle <= -math.pi:
            angle += 2*math.pi
        return angle

    def exec_path(self, curr_pose, end_pose, lateral_dist):
        """
        Generuje sterowanie pojazdem do osiągnięcia end_pose. Przykładowe, póki co nie działa.
        """
        x, y, yaw   = curr_pose
        x_e, y_e, _ = end_pose

        # Oblicz kąt do punktu docelowego i błąd yaw
        target_yaw = math.atan2(y_e - y, x_e - x)
        yaw_err = self.normalize_angle(target_yaw - yaw)

        # Odległość do celu
        dist_forward = math.hypot(x_e - x, y_e - y)

        # Regulator PD dla yaw
        steer = self.Kp * yaw_err + self.Kd * (yaw_err - self.prev_yaw_err)
        self.prev_yaw_err = yaw_err

        # Ustaw kąty sterowania i prędkość
        self.driver.setSteeringAngle(steer)
        self.driver.setCruisingSpeed(C.MAX_SPEED)

        # Dodatkowa korekcja boczna (trzymanie od krawężnika)
        deltalat = lateral_dist - (C.CAR_WIDTH/2 + 0.1)
        steer += self.Kl * deltalat
        self.driver.setSteeringAngle(steer)

    def _make_spot(self, start, end):
        """
        Tworzy opis miejsca parkingowego na podstawie pozycji start/end.
        """
        sp_len = end[0] - start[0]
        if sp_len < self.min_width:
            return None

        # Szerokość miejsca na podstawie różnicy odczytów sonaru
        sp_wid = ((self.dist_start_far - self.dist_start_cl) +
                  (self.dist_end_far   - self.dist_end_cl)) / 2

        # Środek luki parkingowej
        sp_cen_x = start[0] + sp_len / 2
        sp_cen_y = start[1] + (self.dist_start_cl + self.dist_start_far) / 2

        # Ustawienie w zależności od strony parkowania
        if self.side == "left":
            sen_pos = [3.515873,  0.865199,  90]
        elif self.side == "right":
            sen_pos = [3.515873, -0.865199, -90]

        # Punkt końcowy manewru
        x_end = sp_cen_x + sen_pos[0] - 1.425
        y_end = sp_cen_y + sen_pos[1]

        return [x_end, y_end, start[2]]

    def get_spots(self):
        """Zwraca listę wykrytych miejsc parkingowych."""
        return self.spots

