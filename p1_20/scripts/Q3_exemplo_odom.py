#! /usr/bin/env python
# -*- coding:utf-8 -*-

# Sugerimos rodar com:
# roslaunch turtlebot3_gazebo  turtlebot3_empty_world.launch 


from __future__ import print_function, division
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3
import math
import time
from tf import transformations


x = None
y = None

theta = -1


contador = 0
pula = 50

def recebe_odometria(data):
    global x
    global y
    global contador
    global theta

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    quat = data.pose.pose.orientation
    lista = [quat.x, quat.y, quat.z, quat.w]
    angulos = np.degrees(transformations.euler_from_quaternion(lista))    

    if contador % pula == 0:
        print("Posicao (x,y)  ({:.2f} , {:.2f}) + angulo {:.2f}".format(x, y,angulos[2]))
    contador = contador + 1
    theta = np.radians(angulos[2])

if __name__=="__main__":

    rospy.init_node("exemplo_odom")

    t0 = rospy.get_rostime()


    pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )

    ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)

    while not rospy.is_shutdown():

        vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        pub.publish(vel)
        rospy.sleep(0.1)

        print("t0", t0)
        if t0.nsecs == 0:
            t0 = rospy.get_rostime()
            print("waiting for timer")
            continue        
        t1 = rospy.get_rostime()
        elapsed = (t1 - t0)
        print("Passaram ", elapsed.secs, " segundos")

        if elapsed.secs > 30:

            # Obter theta (da odom)
            print(theta)
            # calcular alpha
            alpha = math.atan2(x,y)

            #  Girar a direita 90 + alpha + theta

            w = 0.4

            ang = math.pi/2 + alpha  + theta

            tempo = ang/w

            vel = Twist(Vector3(0,0,0), Vector3(0,0,-w))
            pub.publish(vel)
            rospy.sleep(tempo)

            # andar a hipotenusa
            h = math.sqrt(math.pow(x, 2) + math.pow(y,2))

            v = 0.4
            tempo = h/v

            vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
            pub.publish(vel)
            rospy.sleep(tempo)            
            
            zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
            pub.publish(zero) 
            rospy.sleep(1.0)  

            print("Sucesso")

        rospy.sleep(0.5)
