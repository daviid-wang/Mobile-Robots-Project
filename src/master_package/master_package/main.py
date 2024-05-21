import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy

class ExplorerController(Node):
    def __init__(self):
        super().__init__('explorer_controller')
        self.publisher = self.create_publisher(Bool, '/automatic', 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10) #subscribing yo yt joy, ud dont know what the joy is publishing 
        self.is_automatic = False  # Variable to track the automatic mode
        self.prev_x_state = False
        self.prev_o_state = False

    def joy_callback(self, msg):
        curr_button_x = msg.buttons[2] == 1 # X  
        curr_button_o = msg.buttons[1] == 1 # O
        
        if curr_button_x and not self.prev_x_state:
            self.is_automatic = True #auto mode
            self.publish_mode(self.is_automatic)
        #self.pre_button_state = curr_button_state
        
        if curr_button_o and not self.prev_o_state:
            self.is_automatic = False
            self.publish_mode(self.is_automatic)
        
        self.prev_x_state = curr_button_x
        self.prev_o_state = curr_button_o
        
        #dead man swtich in auto mode
        if self.is_automatic:
            r_2 = msg.axes[5] #rigt
            l_2 = msg.axes[4] #left 
            if r_2 < 0.5 and l_2 < 0.5:
                self.is_automatic = False
                self.publish_mode(self.is_automatic)
        
        # # Assuming that msg.axes[0] corresponds to your dead man switch joystick value
        # if msg.axes[2] > 0.5:  # Assuming > 0.5 means dead man switch is activated
        #     if not self.is_automatic:  # Start explorer.py if not already started
        #         self.is_automatic = True
        #         self.publish_mode(True)
        # else:
        #     if self.is_automatic:  # Stop explorer.py if currently running
        #         self.is_automatic = False
        #         self.publish_mode(False)

    def publish_mode(self, mode):
        msg = Bool()
        msg.data = mode
        self.publisher.publish(msg)
        #self.auto_callback(msg) #!Still broken
    
    def auto_callback(self, msg):
        if msg.data:
            print("Auto Mode ON")
        else:
            print("Auto Mode OFF")

def main(args=None):
    rclpy.init(args=args)
    explorer_controller = ExplorerController()
    rclpy.spin(explorer_controller)
    explorer_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
