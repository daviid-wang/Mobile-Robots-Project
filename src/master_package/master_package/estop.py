# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Bool
# from sensor_msgs.msg import Joy
# from geometry_msgs.msg import Twist

# class ExplorerController(Node):
#     def __init__(self):
#         super().__init__('explorer_controller')
#         self.publisher = self.create_publisher(Bool, '/automatic', 10)
#         self.publisher_cmd_velo = self.create_publisher(Twist,'/cmd_vel', 10)
#         self.subscription_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
#         self.subscription_twist = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10) #subscribing yo yt joy, ud dont know what the joy is publishing 
#         self.is_automatic = False  # Variable to track the automatic mode
#         self.e_stop = False
#         self.prev_x_state = False
#         self.prev_o_state = False
#         self.process_cmd_vel = False

#     def joy_callback(self, msg):
#         curr_button_x = msg.buttons[2] == 1 # X  
#         curr_button_o = msg.buttons[1] == 1 # O
        
#         #auto mode
#         if curr_button_x and not self.prev_x_state:
#             self.is_automatic = True #auto mode
#             self.process_cmd_vel = False 
#             self.publish_mode(self.is_automatic)
#             self.auto_callback(Bool(data=self.is_automatic))
#         #self.pre_button_state = curr_button_state
        
#         #manual mode
#         if curr_button_o and not self.prev_o_state:
#             self.is_automatic = False 
#             self.process_cmd_vel = True 
#             self.publish_mode(self.is_automatic)
#             self.auto_callback(Bool(data=self.is_automatic))
#             self.wheel_halt() 
        
#         self.prev_x_state = curr_button_x
#         self.prev_o_state = curr_button_o
        
#         #dead man swtich in auto mode
#         if self.is_automatic:
#             r_2 = msg.axes[5] #rigt
#             l_2 = msg.axes[4] #left 
#             if (r_2 < -0.5 or l_2 < - 0.5) :
#                 self.get_logger().info("ESTOP INITIATED :O:O:O:O")
#                 self.e_stop = True
#                 self.is_automatic = False
#                 self.process_cmd_vel = False
#                 self.publish_mode(self.is_automatic)
#                 self.wheel_halt()
#             elif (r_2 >= - 0.5 or l_2 >= - 0.5) and self.e_stop:
#                 self.get_logger().info("ESTOP DEACTIVATED!!!! ;)")    
#                 self.e_stop = False
#                 self.process_cmd_vel = True
   
#     # def cmd_vel_callback(self, msg):
#     #     if self.process_cmd_vel and self.is_automatic:
#     #         pass   
#     #     elif not self.process_cmd_vel:
#     #         self.wheel_halt()
    
#     def cmd_vel_callback(self, msg):
#         if self.process_cmd_vel and not self.is_automatic and not self.e_stop:
#             self.publisher_cmd_velo.publish(msg)
#             #pass   
#         else:
#             self.wheel_halt()
            
#     def publish_mode(self, mode):
#         msg = Bool()
#         msg.data = mode
#         self.publisher.publish(msg)
#         #self.auto_callback(msg)
        
#     def auto_callback(self, msg):
#         if msg.data:
#             self.get_logger().info("Auto Mode ON")
#         else:
#             self.get_logger().info("Auto Mode OFF")
            
#     def wheel_halt(self):
#         msg_halt = Twist()
#         msg_halt.linear.x = 0.0
#         msg_halt.linear.y= 0.0
#         msg_halt.linear.z = 0.0
#         msg_halt.angular.x = 0.0
#         msg_halt.angular.y = 0.0
#         msg_halt.angular.z = 0.0
#         self.publisher_cmd_velo.publish(msg_halt)

# def main(args=None):
#     rclpy.init(args=args)
#     explorer_controller = ExplorerController()
#     rclpy.spin(explorer_controller)
#     explorer_controller.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class ExplorerController(Node):
    def __init__(self):
        super().__init__('explorer_controller')
        self.publisher = self.create_publisher(Bool, '/automatic', 10)
        self.publisher_cmd_velo = self.create_publisher(Twist,'/cmd_vel', 10)
        self.subscription_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.subscription_twist = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10) #subscribing yo yt joy, ud dont know what the joy is publishing 
        self.subscription_movement = self.create_subscription(Bool, '/motion_detected', self.movement_callback, 10)
        self.is_automatic = False  # Variable to track the automatic mode
        self.e_stop = False
        self.prev_x_state = False
        self.prev_o_state = False
        self.process_cmd_vel = False
        self.movement_detected = False

    def joy_callback(self, msg):
        curr_button_x = msg.buttons[2] == 1 # X  
        curr_button_o = msg.buttons[1] == 1 # O
        
        #auto mode
        if curr_button_x and not self.prev_x_state:
            self.is_automatic = True #auto mode
            #self.process_cmd_vel = False 
            self.publish_mode(self.is_automatic)
            self.auto_callback(Bool(data=self.is_automatic))
            if not self.movement_detected:
                self.wheel_start() #wheels start in auto mode
        #self.pre_button_state = curr_button_state
        
        #manual mode
        if curr_button_o and not self.prev_o_state:
            self.is_automatic = False 
            self.process_cmd_vel = True 
            self.publish_mode(self.is_automatic)
            self.auto_callback(Bool(data=self.is_automatic))
            self.wheel_halt() 
        
        self.prev_x_state = curr_button_x
        self.prev_o_state = curr_button_o
        
        #dead man swtich in auto mode
        if self.is_automatic:
            r_2 = msg.axes[5] #rigt
            l_2 = msg.axes[4] #left 
            if (r_2 < -0.5 or l_2 < - 0.5) :
                self.get_logger().info("ESTOP INITIATED :O:O:O:O")
                self.e_stop = True
                self.is_automatic = False
                #self.process_cmd_vel = False
                self.publish_mode(self.is_automatic)
                self.wheel_halt()
            elif (r_2 >= - 0.5 or l_2 >= - 0.5) and self.e_stop:
                self.get_logger().info("ESTOP DEACTIVATED!!!! ;)")    
                self.e_stop = False
                #self.process_cmd_vel = True
                self.is_automatic = True
                if not self.movement_detected:
                    self.wheel_start()
   
    # def cmd_vel_callback(self, msg):
    #     if self.process_cmd_vel and self.is_automatic:
    #         pass   
    #     elif not self.process_cmd_vel:
    #         self.wheel_halt()
    
    def cmd_vel_callback(self, msg):
        if self.process_cmd_vel and not self.is_automatic and not self.e_stop:
            #self.get_logger().info("MANUAL MODE!!! PROCESSING CMD_VEL")
            self.publisher_cmd_velo.publish(msg)
            #pass   
        else:
            #self.get_logger().info("cmd_callback: wheel_halt() has been called!!")
            self.wheel_halt()
            
    def movement_callback(self,msg):
        if msg.data and self.is_automatic:
            self.get_logger().info("MOVEMENT DETECTED WHEELS STOPPING..")
            self.movement_detected = True
            self.wheel_halt()
        elif not msg.data and self.is_automatic:
            self.get_logger().info("NO MOVEMENT DETECTED, CARRYING ON......")
            self.movement_detected = False
            self.wheel_start()
                
    def publish_mode(self, mode):
        msg = Bool()
        msg.data = mode
        self.publisher.publish(msg)
        #self.auto_callback(msg)
        
    def auto_callback(self, msg):
        if msg.data:
            self.get_logger().info("Auto Mode ON")
        else:
            self.get_logger().info("Auto Mode OFF")
            
    def wheel_start(self):
        #self.get_logger().info("starting wheels........")
        msg_start = Twist()
        msg_start.linear.x = 0.5
        msg_start.linear.y= 0.0
        msg_start.linear.z = 0.0
        msg_start.angular.x = 0.0
        msg_start.angular.y = 0.0
        msg_start.angular.z = 0.0
        self.publisher_cmd_velo.publish(msg_start)
        
    def wheel_halt(self):
        #self.get_logger().info("halting wheels........")
        msg_halt = Twist()
        msg_halt.linear.x = 0.0
        msg_halt.linear.y= 0.0
        msg_halt.linear.z = 0.0
        msg_halt.angular.x = 0.0
        msg_halt.angular.y = 0.0
        msg_halt.angular.z = 0.0
        self.publisher_cmd_velo.publish(msg_halt)

def main(args=None):
    rclpy.init(args=args)
    explorer_controller = ExplorerController()
    rclpy.spin(explorer_controller)
    explorer_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




