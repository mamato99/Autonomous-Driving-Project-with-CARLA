# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
import math
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._debug = True

        #Parameters for bike overtake
        self.lane_width = 3

        #Parameters for obstacle overtake
        self.lateral_controller = self._local_planner._vehicle_controller._lat_controller

        #Parameters for vehicle overtake
        self.traffic_accident = False

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        if self._behavior.overtake_counter > 0 :
            self._behavior.overtake_counter -= 1
            if self._behavior.overtake_counter == 0:
                self.lateral_controller.set_offset(0) 
                self.traffic_accident = False

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    
#---------------------------------------------  
    def stop_manager(self):
        """
        This method is in charge of behaviors for stop signs.
        """
        stop_list = []
        affected, _ = self._affected_by_stop_sign(stop_list, 5)

        return affected

#--------------------------------------------------------------
    def _move_waypoint(self, num_wp):
        """"
        This method is in charge of moving waypoints on the left lane for the overtake procedure

            :param num_wp: number of waypoints to move
        """
        plan = self._local_planner.get_plan()
        new_plan = []
        
        w = self._local_planner.get_incoming_waypoint_and_direction(0)[0]
        for i in range(num_wp):
            w = w.next(1)[0]
            new_plan.append((w.get_left_lane(), RoadOption.LANEFOLLOW))
            plan.popleft()
        
        for i in plan:
            new_plan.append(i)
        
        self._local_planner.set_global_plan(new_plan)

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                print("new_vehicle_state: " + str(new_vehicle_state) + "vehicle: " + str(vehicle) + "distance: " + str(distance))
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                print("new_vehicle_state: " + str(new_vehicle_state) + "vehicle: " + str(vehicle) + "distance: " + str(distance))
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, ego_waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtake chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
       
        vehicle_list = self.sort_vehicle_list(self._world.get_actors().filter("vehicle.*.*"), ego_waypoint, 60) 

        # Management of the overtake behaviour
        traffic_accident_list = self.sort_vehicle_list(self._world.get_actors().filter("static.prop.warningaccident"), ego_waypoint, 10)
        if len(traffic_accident_list) > 0:
            #This traffic sign report an accident later
            self.traffic_accident = True

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance_vehicle = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance_vehicle = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            
            # Check for vehicle in front of the ego vehicle
            vehicle_state, vehicle, distance_vehicle = self._vehicle_obstacle_detected(
                vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=90)
            
            # If we are in the overtake or tailgating behaviour we exit from the function
            if self._behavior.overtake_counter > 0:
                return False, None, distance_vehicle

            # Section for the overtake
            if vehicle is not None:
                target_transform = vehicle.get_transform()
                target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
                # Case when there is an accident with one or more cars in front of the ego vehicle
                if str(target_wpt.lane_type) == "Shoulder"\
                        and self._direction == RoadOption.LANEFOLLOW \
                        and not ego_waypoint.is_junction \
                        and self._behavior.overtake_counter == 0 \
                        and vehicle.attributes['number_of_wheels'] != '2' \
                        and self.traffic_accident:
                    
                    num_of_vehicles, distance_far_vehicle = self.compute_parameters_for_crashed_vehicles(vehicle_list, ego_waypoint)
                    self._overtake_vehicles(distance_vehicle, distance_far_vehicle, num_of_vehicles, ego_waypoint)

                    return vehicle_state, vehicle, distance_vehicle
                # Case when there is a bycicle in front of the ego vehicle
                elif vehicle.attributes['number_of_wheels'] == '2':
                    self._overtake_bicycles(distance_vehicle, ego_waypoint, target_wpt)

                    return vehicle_state, vehicle, distance_vehicle
                # Case when there is a walker behind the shoulder car
                elif str(target_wpt.lane_type) == "Shoulder" and not self.traffic_accident:

                    return False, None, -1
                # Case when there is a car to follow
                else:

                    return vehicle_state, vehicle, distance_vehicle
    
                
        return vehicle_state, vehicle, distance_vehicle

    def compute_parameters_for_crashed_vehicles(self, vehicle_list, ego_waypoint):
        distance_far_vehicle = 0 
        num_of_vehicles = 0 
        #Compute how many vehicle there are in front of the ego vehicle
        for vehicle in vehicle_list:
            target_transform = vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            if str(target_wpt.lane_type) == "Shoulder":
                new_dist = compute_distance(target_transform.location, ego_waypoint.transform.location)
                if new_dist > distance_far_vehicle:
                    distance_far_vehicle = new_dist
                num_of_vehicles += 1

        return num_of_vehicles, distance_far_vehicle
    
    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """
        walker_list = self.sort_vehicle_list(self._world.get_actors().filter("walker.pedestrian.*"), waypoint, self._speed_limit / 2)

        if len(walker_list) == 0:
            return (False, None, 0)
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

#-------------------------------------------------------------------------------
    def _overtake_bicycles(self, distance_front_vehicle, ego_waypoint, bike_waypoint):
        
        vehicle_list = self.sort_vehicle_list(self._world.get_actors().filter("*vehicle*"), ego_waypoint, 150)
        new_vehicle_list = []
        bike_list = []
        dist_far_bycicle = 0
        for v in vehicle_list:
            if v.attributes['number_of_wheels'] != '2':
                new_vehicle_list.append(v)
            else:
                bike_list.append(v)
                new_dist = compute_distance(v.get_location(), self._vehicle.get_location())
                if new_dist > dist_far_bycicle:
                    dist_far_bycicle = new_dist

        # Check for vehicles in the other lane
        side_vehicle_state, _ , distance_side_vehicle = self._vehicle_obstacle_detected(new_vehicle_list, max(
                self._behavior.min_proximity_threshold, 150), low_angle_th=0, up_angle_th=60, lane_offset=-2)
        
        # Compute the distance between two bycicles
        dist_btw_bikes = 0
        if len(bike_list) >= 2:
            dist_btw_bikes = compute_distance(bike_list[0].get_location(), bike_list[1].get_location()) - self._vehicle.bounding_box.extent.x/4 * 2 

        # Check the angle of the road for evaluate the overtake
        angle = 0
        angle = self.compute_angle()
        
        if side_vehicle_state:
            if(self._debug):
                print("Ci sta un veicolo non posso sorpassare a sinistra")
        
        if(self._debug):
            print("Distanza del veicolo dall'altra corsia " + str(distance_front_vehicle))
            print("Distanza del veicolo dall'ego_vehicle nella mia corsia: " + str(distance_front_vehicle))
        
        # if the size of the roadway permits it, it allows overtake the bycicles
        if bike_waypoint.lane_width > 3 and distance_front_vehicle < 8:
            self._behavior.overtake_counter = 80
            self.lateral_controller.set_offset(-self.lane_width/2 + 0.3)
            return
        
        look_ahead_distance = 35 + dist_far_bycicle
        counter = 70
        # change the overtaking parameters allowing the return if the distance between the bicycles permits it
        if dist_btw_bikes > self._vehicle.bounding_box.extent.x * 3 and self._speed != 0:
            counter = 50
            look_ahead_distance = self._speed_limit + distance_front_vehicle
        elif self._speed != 0:
            counter = 80
            look_ahead_distance = 1.3 * self._speed_limit + dist_far_bycicle

        if (not side_vehicle_state or \
            (distance_side_vehicle > look_ahead_distance)) \
            and distance_front_vehicle < 8 \
            and angle > 173 \
            and self._behavior.overtake_counter == 0:

            self._behavior.overtake_counter = counter
            self.lateral_controller.set_offset(-self.lane_width/2 + 0.3)



    def dist(self, v, ego_waypoint): 
        distance_vehicle = compute_distance(v.get_location(), ego_waypoint.transform.location)
        distance_vehicle = distance_vehicle - max(
                v.bounding_box.extent.y, v.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
        return distance_vehicle 

    def sort_vehicle_list(self, vehicle_list, ego_waypoint, distance):  
        """
        This function has the task of returning an ordered list of objects based on the distance passed as input
        """
          
        vehicle_dict = {}  
        for v in vehicle_list:
            dist = self.dist(v, ego_waypoint) 
            if dist < distance and v.id != self._vehicle.id and dist > 0:
                vehicle_dict[v] = dist
        sorted_dict = sorted(vehicle_dict.items(), key=lambda x: x[1])
        vehicle_list= []
        for i in range(len(sorted_dict)):
            vehicle_list.append(sorted_dict[i][0])
        return vehicle_list
    
    def compute_angle(self):
        w0 = self._local_planner.get_incoming_waypoint_and_direction(0)[0]
        w1 = self._local_planner.get_incoming_waypoint_and_direction(5)[0]
        w2 = self._local_planner.get_incoming_waypoint_and_direction(10)[0]
        p1 = (w0.transform.location.x,w0.transform.location.y)
        p2 = (w1.transform.location.x, w1.transform.location.y)
        p3 = (w2.transform.location.x, w2.transform.location.y)
        
    # Compute the distances between points
        a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        b = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        c = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        # Compute the angle using the cosine theorem
        angle = math.degrees(math.acos((a**2 + + c**2 - b**2) / (2 * a * c)))
        return angle
            
    def _overtake_static_obstacle(self, ego_waypoint, obstacle_list, cone_list):

            vehicle_list = self.sort_vehicle_list(self._world.get_actors().filter("*vehicle*"), ego_waypoint, 100)
            # Check vehicles in the other lane of the ego vehicle
            side_vehicle_state , _ , distance_side_vehicle  = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, 100), low_angle_th=0, up_angle_th=45, lane_offset=-2)
                       
            # Check the distance from the obstacle in front of the ego vehicle
            _ , obstacle ,distance_obstacle  = self._vehicle_obstacle_detected(obstacle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=30, lane_offset=0)
        
            if side_vehicle_state:
                if(self._debug):
                    print("Ci sta un veicolo non posso sorpassare a sinistra")
            
            if(self._debug):
                print("Distanza dell'oggetto dall'ego_vehicle nella mia corsia: " + str(distance_obstacle))

            dist_c = 0
            if len(cone_list) > 0:
                dist_c = compute_distance(cone_list[-1].get_location(), ego_waypoint.transform.location)
                
            if (not side_vehicle_state or \
                (distance_side_vehicle > self._speed_limit + dist_c)) \
                and distance_obstacle < (self._speed_limit / 4) \
                and self._behavior.overtake_counter == 0:
                if(self._debug):
                    print("Overtaking to the left!")

                self._behavior.overtake_counter = 200
                self._move_waypoint(int(dist_c))
                return None, -1
            else:
                return obstacle, distance_obstacle         
    
    def _overtake_vehicles(self, distance_front_vehicle, distance_far_vehicle, num_of_vehicle, ego_waypoint):

        if(self._debug):
            print('overtake_car called')
        
        # Compute a new vehicle list to see more vehicles
        vehicle_list = self.sort_vehicle_list(self._world.get_actors().filter("*vehicle*"), ego_waypoint, 150)
        new_vehicle_list = []
        # Remove all the crashed vehicles from the vehicle list
        for vehicle in vehicle_list:
            target_transform = vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            if str(target_wpt.lane_type) != "Shoulder":
                new_vehicle_list.append(vehicle)  
                
        # Check for vehicles in the other lane
        side_vehicle_state , side_vehicle , distance_side_vehicle  = self._vehicle_obstacle_detected(new_vehicle_list, max(
                self._behavior.min_proximity_threshold, 150), low_angle_th=0, up_angle_th=45, lane_offset=-2)
        
        if side_vehicle_state:
            if(self._debug):
                print("Ci sta un veicolo non posso sorpassare a sinistra")
        
        if(self._debug):
            print("Distanza del veicolo dall'ego_vehicle nella mia corsia: " + str(distance_front_vehicle))

        # Modify the look ahead distance based on the number of vehicles
        if num_of_vehicle > 1:
            look_ahead_distance = 35 * num_of_vehicle
        else:
            look_ahead_distance = 1.5 * self._speed_limit
       
        if (not side_vehicle_state or \
            (distance_side_vehicle > look_ahead_distance)) \
            and distance_front_vehicle < (self._speed_limit / 4) \
            and self._behavior.overtake_counter == 0:
            if(self._debug):
                print("Overtaking to the left! ")

            self._behavior.overtake_counter = 200
            self._move_waypoint(int(distance_far_vehicle))

    def lane_narrowing_and_static_obstacle(self, ego_vehicle_wp):
        cone_list = self.sort_vehicle_list(self._world.get_actors().filter('static.prop.constructioncone'), ego_vehicle_wp, 25)
        traffic_warning_list = self.sort_vehicle_list(self._world.get_actors().filter('static.prop.trafficwarning'), ego_vehicle_wp, 30)
    
        if len(cone_list) != 0 and len(traffic_warning_list) == 0 and self._behavior.overtake_counter == 0:
            lane_width = ego_vehicle_wp.lane_width
            self.lateral_controller.set_offset(lane_width/2 - 0.25)
            self._behavior.overtake_counter = len(cone_list) * 5
            return None, -1
        elif len(traffic_warning_list) > 0 and \
            self._direction == RoadOption.LANEFOLLOW \
            and not ego_vehicle_wp.is_junction \
            and self._behavior.overtake_counter == 0:
            obstacle, distance = self._overtake_static_obstacle(ego_vehicle_wp, traffic_warning_list, cone_list)
            return obstacle, distance
        else:
            return None, -1
#---------------------------------------------------------------------------------
    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()

        control = None

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
    
        waypoints_debug = []
        for i in range(6):
            waypoints_debug.append(self._local_planner.get_incoming_waypoint_and_direction(i+1))
        for waypoint_debug in waypoints_debug:
            self._world.debug.draw_string(waypoint_debug[0].transform.location, 'O', draw_shadow=False,
                                             color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                             persistent_lines=True)
            
        # 1: Red lights and stops behavior
        if  self.traffic_light_manager() or self.stop_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2 Static obstacle behavior
        obstacle, distance = self.lane_narrowing_and_static_obstacle(ego_vehicle_wp)
        if obstacle is not None and distance < 12:
            return self.emergency_stop()
        
        # 2.3: Car and bikes behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        
        if vehicle_state and self._behavior.overtake_counter == 0:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            # Emergency brake if the car is very close.
            if vehicle.attributes['number_of_wheels'] == '2' and distance > 2 and distance < self._speed_limit/3:
                self._local_planner.set_speed(20)
                return self._local_planner.run_step()
            else:
                if distance < self._behavior.braking_distance:
                    return self.emergency_stop()
                else:
                    control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):

            vehicle_list = self.sort_vehicle_list(self._world.get_actors().filter("vehicle.*.*"), ego_vehicle_wp, 70) 
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                self._behavior.min_proximity_threshold, 70))
            if not vehicle_state or distance > self._speed_limit / 2:
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 5])
                self._local_planner.set_speed(target_speed)
                return self._local_planner.run_step(debug=debug)
            else:
                light_state = vehicle.get_light_state()
                right_blinker = carla.VehicleLightState.NONE
                right_blinker |= carla.VehicleLightState.RightBlinker
                bit_rb_presente = light_state & right_blinker != 0
                #The RightBlinker is encoded on 8 bit
                if bit_rb_presente:
                    target_speed = min([
                        self._behavior.max_speed,
                        self._speed_limit - 5])
                    self._local_planner.set_speed(target_speed)
                    return self._local_planner.run_step()
                else:
                    return self.emergency_stop()

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
               self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
